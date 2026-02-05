"""
Gradio UI for TRELLIS.2 Multi-View 3D Generation

Supports uploading multiple view images and tweaking all sampler parameters.
Features live logging and quality metrics.
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import gradio as gr
from datetime import datetime
import shutil
import cv2
import time
from typing import *
import threading
from io import StringIO
import torch
import numpy as np
from PIL import Image
import base64
import io
from trellis2.pipelines import Trellis2MultiViewPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
from trellis2.utils.quality_metrics import compute_quality_metrics, format_metrics_report
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_multiview')
MAX_IMAGES = 8  # Maximum number of input views supported


def make_progress_bar(percent: int, status: str) -> str:
    """Generate Markdown progress indicator (better streaming support than HTML)."""
    filled = int(percent / 5)  # 20 chars total
    empty = 20 - filled
    bar = "█" * filled + "░" * empty
    return f"**{status}**\n\n`[{bar}]` {percent}%"


class ConsoleCapture:
    """Thread-safe stdout/stderr capture for Gradio."""

    def __init__(self):
        self.buffer = StringIO()
        self._lock = threading.Lock()
        self._original_stdout = None
        self._original_stderr = None

    def start(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def stop(self):
        if self._original_stdout:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._original_stdout = None
            self._original_stderr = None

    def write(self, text):
        with self._lock:
            self.buffer.write(text)
        # Also write to original stdout
        if self._original_stdout:
            self._original_stdout.write(text)

    def flush(self):
        if self._original_stdout:
            self._original_stdout.flush()

    def get_output(self):
        with self._lock:
            return self.buffer.getvalue()

    def clear(self):
        with self._lock:
            self.buffer = StringIO()


class GenerationLogger:
    """Helper class for formatted generation logging."""

    def __init__(self):
        self.logs = []
        self.start_time = None

    def reset(self):
        self.logs = []
        self.start_time = time.time()

    def log(self, message: str) -> str:
        """Add a timestamped log message and return full log string."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.logs.append(entry)
        return "\n".join(self.logs)

    def elapsed(self) -> float:
        """Get elapsed time since reset."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def format_elapsed(self) -> str:
        """Format elapsed time as string."""
        return f"{self.elapsed():.1f}s"


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def preprocess_images(images: List[Image.Image]) -> List[Image.Image]:
    """Preprocess all input images."""
    if images is None:
        return []
    processed = []
    for img in images:
        if img is not None:
            processed.append(pipeline.preprocess_image(img))
    return processed


def generate_3d(
    images: List[Image.Image],
    mode: str,
    pipeline_type: str,
    seed: int,
    # Sparse Structure params
    ss_steps: int,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_rescale_t: float,
    # Shape SLat params
    shape_steps: int,
    shape_guidance_strength: float,
    shape_guidance_rescale: float,
    shape_rescale_t: float,
    # Texture SLat params
    tex_steps: int,
    tex_guidance_strength: float,
    tex_guidance_rescale: float,
    tex_rescale_t: float,
    # Export params
    decimation_target: int,
    texture_size: int,
    # Metrics option
    compute_metrics: bool,
    req: gr.Request,
    progress=gr.Progress(),
):
    """Generate 3D model from multiple view images with live logging."""
    logger = GenerationLogger()
    logger.reset()
    console = ConsoleCapture()
    console.start()

    # Filter out None images
    valid_images = [img for img in images if img is not None]

    if len(valid_images) < 1:
        raise gr.Error("Please upload at least one image.")

    if len(valid_images) == 1:
        gr.Warning("Only 1 image provided. Multi-view works best with 2+ images.")

    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

    # Map pipeline type names
    pipeline_type_map = {
        "512": "512",
        "1024 (Cascade)": "1024_cascade",
        "1024 (Direct)": "1024",
        "1536 (Cascade)": "1536_cascade",
    }
    pt = pipeline_type_map[pipeline_type]

    # Determine cascade info for logging
    is_cascade = 'cascade' in pt
    shape_passes = 2 if is_cascade else 1

    try:
        # Yield: Starting
        # Outputs: logs, progress, console, video, model, metrics_report, dino, lpips, ssim (9 values)
        # Note: download_btn is updated separately via .then() to avoid Gradio 6 streaming issues
        yield (
            logger.log(f"Starting generation with {len(valid_images)} view(s)..."),
            make_progress_bar(0, "Initializing..."),
            console.get_output(),
            None, None,  # video, model
            None, None, None, None,  # metrics report, dino, lpips, ssim
        )

        # === Stage 1: Sparse Structure ===
        yield (
            logger.log(f"Stage 1/4: Sampling Sparse Structure ({ss_steps} steps)..."),
            make_progress_bar(5, "Stage 1/4: Sparse Structure..."),
            console.get_output(),
            None, None, None, None, None, None,
        )

        stage_start = time.time()

        # Get conditioning
        cond_512 = pipeline.get_cond_multi(valid_images, 512)
        cond_1024 = pipeline.get_cond_multi(valid_images, 1024) if pt != '512' else None

        torch.manual_seed(seed)
        num_images = len(valid_images)

        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pt]

        with pipeline.inject_sampler_multi_image(
            pipeline.sparse_structure_sampler, num_images, ss_steps, mode.lower()
        ):
            coords = pipeline.sample_sparse_structure(
                cond_512, ss_res, 1,
                {
                    "steps": ss_steps,
                    "guidance_strength": ss_guidance_strength,
                    "guidance_rescale": ss_guidance_rescale,
                    "rescale_t": ss_rescale_t,
                }
            )

        stage_time = time.time() - stage_start
        yield (
            logger.log(f"Stage 1/4: Complete ({stage_time:.1f}s)"),
            make_progress_bar(25, "Stage 1/4: Complete ✓"),
            console.get_output(),
            None, None, None, None, None, None,
        )

        # === Stage 2: Shape Latent ===
        stage_desc = f"Stage 2/4: Sampling Shape Latent ({shape_steps} steps"
        if is_cascade:
            stage_desc += ", 2-pass cascade"
        stage_desc += ")..."

        yield (
            logger.log(stage_desc),
            make_progress_bar(25, "Stage 2/4: Shape Latent..."),
            console.get_output(),
            None, None, None, None, None, None,
        )

        stage_start = time.time()

        shape_sampler_params = {
            "steps": shape_steps,
            "guidance_strength": shape_guidance_strength,
            "guidance_rescale": shape_guidance_rescale,
            "rescale_t": shape_rescale_t,
        }

        if pt == '512':
            with pipeline.inject_sampler_multi_image(
                pipeline.shape_slat_sampler, num_images, shape_steps, mode.lower()
            ):
                shape_slat = pipeline.sample_shape_slat(
                    cond_512, pipeline.models['shape_slat_flow_model_512'],
                    coords, shape_sampler_params
                )
            res = 512

        elif pt == '1024':
            with pipeline.inject_sampler_multi_image(
                pipeline.shape_slat_sampler, num_images, shape_steps, mode.lower()
            ):
                shape_slat = pipeline.sample_shape_slat(
                    cond_1024, pipeline.models['shape_slat_flow_model_1024'],
                    coords, shape_sampler_params
                )
            res = 1024

        elif pt in ['1024_cascade', '1536_cascade']:
            target_res = 1024 if pt == '1024_cascade' else 1536

            with pipeline.inject_sampler_multi_image(
                pipeline.shape_slat_sampler, num_images, shape_steps * 2, mode.lower()
            ):
                shape_slat, res = pipeline.sample_shape_slat_cascade(
                    cond_512, cond_1024,
                    pipeline.models['shape_slat_flow_model_512'],
                    pipeline.models['shape_slat_flow_model_1024'],
                    512, target_res,
                    coords, shape_sampler_params,
                    49152  # max_num_tokens
                )

        stage_time = time.time() - stage_start
        yield (
            logger.log(f"Stage 2/4: Complete ({stage_time:.1f}s)"),
            make_progress_bar(50, "Stage 2/4: Complete ✓"),
            console.get_output(),
            None, None, None, None, None, None,
        )

        # === Stage 3: Texture Latent ===
        yield (
            logger.log(f"Stage 3/4: Sampling Texture Latent ({tex_steps} steps)..."),
            make_progress_bar(50, "Stage 3/4: Texture Latent..."),
            console.get_output(),
            None, None, None, None, None, None,
        )

        stage_start = time.time()

        tex_cond = cond_512 if pt == '512' else cond_1024
        tex_model = pipeline.models['tex_slat_flow_model_512'] if pt == '512' else pipeline.models['tex_slat_flow_model_1024']

        with pipeline.inject_sampler_multi_image(
            pipeline.tex_slat_sampler, num_images, tex_steps, mode.lower()
        ):
            tex_slat = pipeline.sample_tex_slat(
                tex_cond, tex_model, shape_slat,
                {
                    "steps": tex_steps,
                    "guidance_strength": tex_guidance_strength,
                    "guidance_rescale": tex_guidance_rescale,
                    "rescale_t": tex_rescale_t,
                }
            )

        stage_time = time.time() - stage_start
        yield (
            logger.log(f"Stage 3/4: Complete ({stage_time:.1f}s)"),
            make_progress_bar(75, "Stage 3/4: Complete ✓"),
            console.get_output(),
            None, None, None, None, None, None,
        )

        # === Stage 4: Decoding & Export ===
        yield (
            logger.log("Stage 4/4: Decoding mesh and rendering..."),
            make_progress_bar(75, "Stage 4/4: Decoding..."),
            console.get_output(),
            None, None, None, None, None, None,
        )

        stage_start = time.time()

        torch.cuda.empty_cache()
        mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]

        # Simplify mesh (nvdiffrast limit)
        mesh.simplify(16777216)

        decode_time = time.time() - stage_start
        yield (
            logger.log(f"Stage 4/4: Mesh decoded ({decode_time:.1f}s)"),
            make_progress_bar(80, "Stage 4/4: Mesh decoded ✓"),
            console.get_output(),
            None, None, None, None, None, None,
        )

        # Render preview video frames
        yield (
            logger.log("Stage 4/4: Rendering preview video..."),
            make_progress_bar(80, "Stage 4/4: Rendering video..."),
            console.get_output(),
            None, None, None, None, None, None,
        )

        render_start = time.time()
        video_frames = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )

        # Save preview video
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
        video_path = os.path.join(user_dir, f'preview_{timestamp}.mp4')

        import imageio
        imageio.mimsave(video_path, video_frames, fps=15)

        render_time = time.time() - render_start
        yield (
            logger.log(f"Stage 4/4: Video rendered ({render_time:.1f}s)"),
            make_progress_bar(85, "Stage 4/4: Video rendered ✓"),
            console.get_output(),
            None, None, None, None, None, None,
        )

        # Export GLB
        yield (
            logger.log("Stage 4/4: Exporting GLB..."),
            make_progress_bar(85, "Stage 4/4: Exporting GLB..."),
            console.get_output(),
            None, None, None, None, None, None,
        )

        export_start = time.time()
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )

        glb_path = os.path.join(user_dir, f'model_{timestamp}.glb')
        glb.export(glb_path, extension_webp=True)

        export_time = time.time() - export_start
        yield (
            logger.log(f"Stage 4/4: GLB exported ({export_time:.1f}s)"),
            make_progress_bar(90, "Stage 4/4: GLB exported ✓"),
            console.get_output(),
            video_path, glb_path,
            None, None, None, None,
        )

        # === Quality Metrics (optional) ===
        metrics_report = ""
        dino_val = None
        lpips_val = None
        ssim_val = None

        print(f"DEBUG: compute_metrics = {compute_metrics}", flush=True)
        if compute_metrics:
            yield (
                logger.log("Computing quality metrics..."),
                make_progress_bar(95, "Computing quality metrics..."),
                console.get_output(),
                video_path, glb_path,
                None, None, None, None,
            )

            metrics_start = time.time()
            try:
                print("DEBUG: Starting quality metrics computation...", flush=True)
                metrics = compute_quality_metrics(
                    valid_images,
                    mesh,
                    pipeline.image_cond_model,
                    render_resolution=512,
                    metric_resolution=256,
                    envmap=envmap,
                )
                metrics_report = format_metrics_report(metrics)
                dino_val = round(metrics['dino_similarity'], 3)
                lpips_val = round(metrics['lpips'], 3)
                ssim_val = round(metrics['ssim'], 3)
                print(f"DEBUG: Metrics computed: DINO={dino_val}, LPIPS={lpips_val}, SSIM={ssim_val}", flush=True)

                metrics_time = time.time() - metrics_start
                yield (
                    logger.log(f"Metrics computed ({metrics_time:.1f}s)"),
                    make_progress_bar(100, "Metrics computed ✓"),
                    console.get_output(),
                    video_path, glb_path,
                    metrics_report, dino_val, lpips_val, ssim_val,
                )
            except Exception as e:
                import traceback
                print(f"DEBUG: Metrics computation FAILED: {str(e)}", flush=True)
                print(f"DEBUG: Full traceback:\n{traceback.format_exc()}", flush=True)
                yield (
                    logger.log(f"Metrics failed: {str(e)}"),
                    make_progress_bar(100, "Metrics failed"),
                    console.get_output(),
                    video_path, glb_path,
                    f"Error computing metrics: {str(e)}", None, None, None,
                )

        # Final summary
        total_time = logger.elapsed()
        final_log = logger.log(f"Generation complete! Total time: {total_time:.1f}s")

        torch.cuda.empty_cache()

        yield (
            final_log,
            make_progress_bar(100, f"Complete! ({total_time:.1f}s)"),
            console.get_output(),
            video_path, glb_path,
            metrics_report, dino_val, lpips_val, ssim_val,
        )
    finally:
        console.stop()


def create_ui():
    with gr.Blocks(title="TRELLIS.2 Multi-View") as demo:
        gr.Markdown("""
        # TRELLIS.2 Multi-View 3D Generation

        Upload multiple views of an object to generate higher-quality 3D models.
        The multi-view approach aggregates predictions from different viewpoints during denoising.

        **Tips:**
        - Use 2-4 views for best results (front, back, left, right)
        - Images should have consistent lighting and background
        - Enable background removal for best results
        """)

        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input Images")

                # Image gallery for multiple uploads
                image_input = gr.Gallery(
                    label="Upload Views (drag & drop multiple images)",
                    columns=4,
                    rows=2,
                    height=300,
                    object_fit="contain",
                    type="pil",
                    allow_preview=True,
                )

                preprocess_btn = gr.Button("Preprocess Images (Remove Background)", variant="secondary")

                gr.Markdown("### Generation Settings")

                with gr.Row():
                    mode = gr.Radio(
                        choices=["Stochastic", "Multidiffusion"],
                        value="Multidiffusion",
                        label="Aggregation Mode",
                        info="Stochastic: Faster, cycles views. Multidiffusion: Slower, averages all views."
                    )

                with gr.Row():
                    pipeline_type = gr.Dropdown(
                        choices=["512", "1024 (Cascade)", "1024 (Direct)", "1536 (Cascade)"],
                        value="1024 (Cascade)",
                        label="Pipeline Type",
                        info="Higher resolution = better quality but slower"
                    )

                with gr.Row():
                    seed = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
                    randomize_seed = gr.Checkbox(label="Randomize", value=False)

                with gr.Row():
                    compute_metrics = gr.Checkbox(
                        label="Compute Quality Metrics",
                        value=True,
                        info="Calculate DINO/LPIPS/SSIM after generation"
                    )

                generate_btn = gr.Button("Generate 3D Model", variant="primary", size="lg")

                # Advanced parameters in accordion
                with gr.Accordion("Advanced: Sparse Structure Sampler", open=False):
                    gr.Markdown("Controls the initial sparse voxel structure generation")
                    with gr.Row():
                        ss_steps = gr.Slider(1, 50, value=12, step=1, label="Steps")
                        ss_guidance_strength = gr.Slider(1.0, 10.0, value=7.5, step=0.1, label="Guidance Strength")
                    with gr.Row():
                        ss_guidance_rescale = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Guidance Rescale")
                        ss_rescale_t = gr.Slider(1.0, 6.0, value=5.0, step=0.1, label="Rescale T")

                with gr.Accordion("Advanced: Shape SLat Sampler", open=False):
                    gr.Markdown("Controls the shape latent generation")
                    with gr.Row():
                        shape_steps = gr.Slider(1, 50, value=12, step=1, label="Steps")
                        shape_guidance_strength = gr.Slider(1.0, 10.0, value=7.5, step=0.1, label="Guidance Strength")
                    with gr.Row():
                        shape_guidance_rescale = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Guidance Rescale")
                        shape_rescale_t = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Rescale T")

                with gr.Accordion("Advanced: Texture SLat Sampler", open=False):
                    gr.Markdown("Controls the texture/material generation")
                    with gr.Row():
                        tex_steps = gr.Slider(1, 50, value=12, step=1, label="Steps")
                        tex_guidance_strength = gr.Slider(1.0, 10.0, value=1.0, step=0.1, label="Guidance Strength")
                    with gr.Row():
                        tex_guidance_rescale = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Guidance Rescale")
                        tex_rescale_t = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Rescale T")

                with gr.Accordion("Export Settings", open=False):
                    decimation_target = gr.Slider(
                        100000, 1000000, value=500000, step=10000,
                        label="Decimation Target (faces)"
                    )
                    texture_size = gr.Slider(
                        1024, 4096, value=2048, step=1024,
                        label="Texture Size"
                    )

            # Right column: Outputs
            with gr.Column(scale=1):
                gr.Markdown("### Output")

                # Logs tabs with Generation Logs and Console Output
                with gr.Tabs():
                    with gr.Tab("Generation Logs"):
                        live_logs = gr.Textbox(
                            label="Stage Progress",
                            interactive=False,
                            lines=10,
                            max_lines=15,
                            autoscroll=True,
                        )
                        progress_bar = gr.Markdown(
                            value=make_progress_bar(0, "Ready"),
                        )
                    with gr.Tab("Console Output"):
                        console_output = gr.Textbox(
                            label="Debug/Console Output",
                            interactive=False,
                            lines=12,
                            max_lines=20,
                            autoscroll=True,
                        )

                with gr.Tabs():
                    with gr.Tab("Video Preview"):
                        video_output = gr.Video(label="360° Preview", height=400)

                    with gr.Tab("3D Model"):
                        model_output = gr.Model3D(
                            label="GLB Model",
                            height=400,
                            clear_color=(0.25, 0.25, 0.25, 1.0),
                        )

                download_btn = gr.DownloadButton(label="Download GLB", variant="secondary")

                # Quality Metrics
                with gr.Accordion("Quality Metrics", open=True):
                    metrics_output = gr.Textbox(
                        label="Metrics Report",
                        interactive=False,
                        lines=10,
                    )
                    with gr.Row():
                        dino_sim = gr.Number(label="DINO Similarity", interactive=False)
                        lpips_val = gr.Number(label="LPIPS (lower=better)", interactive=False)
                        ssim_val = gr.Number(label="SSIM (higher=better)", interactive=False)

        # Example images
        gr.Markdown("### Example: Cyborg Rat (4 views)")
        gr.Markdown("Load example images from `assets/cyborg_rat/` folder")

        def load_example():
            example_dir = "assets/cyborg_rat"
            if os.path.exists(example_dir):
                images = []
                for fname in sorted(os.listdir(example_dir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images.append(Image.open(os.path.join(example_dir, fname)))
                return images
            return []

        load_example_btn = gr.Button("Load Cyborg Rat Example")
        load_example_btn.click(load_example, outputs=[image_input])

        # Event handlers
        demo.load(start_session)
        demo.unload(end_session)

        def preprocess_gallery(images):
            if images is None or len(images) == 0:
                return []
            # Gallery returns list of tuples (image, caption) or just images
            pil_images = []
            for item in images:
                if isinstance(item, tuple):
                    pil_images.append(item[0])
                else:
                    pil_images.append(item)
            return preprocess_images(pil_images)

        preprocess_btn.click(
            preprocess_gallery,
            inputs=[image_input],
            outputs=[image_input],
        )

        def prepare_and_generate(
            images, mode, pipeline_type, randomize, seed,
            ss_steps, ss_guidance_strength, ss_guidance_rescale, ss_rescale_t,
            shape_steps, shape_guidance_strength, shape_guidance_rescale, shape_rescale_t,
            tex_steps, tex_guidance_strength, tex_guidance_rescale, tex_rescale_t,
            decimation_target, texture_size, compute_metrics,
            req: gr.Request,
            progress=gr.Progress(),
        ):
            # Get seed
            actual_seed = get_seed(randomize, seed)

            # Extract PIL images from gallery
            pil_images = []
            if images:
                for item in images:
                    if isinstance(item, tuple):
                        pil_images.append(item[0])
                    else:
                        pil_images.append(item)

            # Preprocess if not already done
            preprocessed = preprocess_images(pil_images)

            # Yield from generator
            yield from generate_3d(
                preprocessed, mode, pipeline_type, actual_seed,
                ss_steps, ss_guidance_strength, ss_guidance_rescale, ss_rescale_t,
                shape_steps, shape_guidance_strength, shape_guidance_rescale, shape_rescale_t,
                tex_steps, tex_guidance_strength, tex_guidance_rescale, tex_rescale_t,
                decimation_target, texture_size, compute_metrics,
                req, progress,
            )

        # Helper to update download button from model path
        def update_download_btn(model_path):
            return model_path

        generate_btn.click(
            prepare_and_generate,
            inputs=[
                image_input, mode, pipeline_type, randomize_seed, seed,
                ss_steps, ss_guidance_strength, ss_guidance_rescale, ss_rescale_t,
                shape_steps, shape_guidance_strength, shape_guidance_rescale, shape_rescale_t,
                tex_steps, tex_guidance_strength, tex_guidance_rescale, tex_rescale_t,
                decimation_target, texture_size, compute_metrics,
            ],
            outputs=[
                live_logs,
                progress_bar,
                console_output,
                video_output, model_output,
                metrics_output, dino_sim, lpips_val, ssim_val,
            ],
        ).then(
            # Update download button after generation completes (avoids Gradio 6 streaming issues)
            update_download_btn,
            inputs=[model_output],
            outputs=[download_btn],
        )

        return demo


# Launch
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--server-name", type=str, default="0.0.0.0", help="Server name/IP to bind to")
    args = parser.parse_args()

    os.makedirs(TMP_DIR, exist_ok=True)

    # Load pipeline
    print("Loading TRELLIS.2 Multi-View Pipeline...")
    pipeline = Trellis2MultiViewPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    # Load environment map for rendering
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    print("Starting Gradio UI...")
    print(f"Server: {args.server_name}:{args.port}")
    if args.share:
        print("Public link will be generated...")

    demo = create_ui()
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        # Azure ML specific settings
        root_path=os.environ.get("GRADIO_ROOT_PATH", ""),
    )
