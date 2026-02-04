"""
Gradio UI for TRELLIS.2 Multi-View 3D Generation

Supports uploading multiple view images and tweaking all sampler parameters.
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
from datetime import datetime
import shutil
import cv2
from typing import *
import torch
import numpy as np
from PIL import Image
import base64
import io
from trellis2.pipelines import Trellis2MultiViewPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
import o_voxel


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_multiview')
MAX_IMAGES = 8  # Maximum number of input views supported


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
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate 3D model from multiple view images."""
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

    # Run multi-view generation
    mesh = pipeline.run_multi_image(
        valid_images,
        seed=seed,
        mode=mode.lower(),
        pipeline_type=pt,
        preprocess_image=False,  # Already preprocessed
        sparse_structure_sampler_params={
            "steps": ss_steps,
            "guidance_strength": ss_guidance_strength,
            "guidance_rescale": ss_guidance_rescale,
            "rescale_t": ss_rescale_t,
        },
        shape_slat_sampler_params={
            "steps": shape_steps,
            "guidance_strength": shape_guidance_strength,
            "guidance_rescale": shape_guidance_rescale,
            "rescale_t": shape_rescale_t,
        },
        tex_slat_sampler_params={
            "steps": tex_steps,
            "guidance_strength": tex_guidance_strength,
            "guidance_rescale": tex_guidance_rescale,
            "rescale_t": tex_rescale_t,
        },
    )[0]

    # Simplify mesh (nvdiffrast limit)
    mesh.simplify(16777216)

    # Render preview video frames
    video_frames = render_utils.make_pbr_vis_frames(
        render_utils.render_video(mesh, envmap=envmap)
    )

    # Save preview video
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    video_path = os.path.join(user_dir, f'preview_{timestamp}.mp4')

    import imageio
    imageio.mimsave(video_path, video_frames, fps=15)

    # Export GLB
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

    torch.cuda.empty_cache()

    # Return info string, video, glb, and download path
    info = f"Generated with {len(valid_images)} view(s), mode={mode}, pipeline={pt}, seed={seed}"
    return info, video_path, glb_path, glb_path


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

                output_info = gr.Textbox(label="Generation Info", interactive=False)

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
            decimation_target, texture_size,
            req: gr.Request,
            progress=gr.Progress(track_tqdm=True),
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

            return generate_3d(
                preprocessed, mode, pipeline_type, actual_seed,
                ss_steps, ss_guidance_strength, ss_guidance_rescale, ss_rescale_t,
                shape_steps, shape_guidance_strength, shape_guidance_rescale, shape_rescale_t,
                tex_steps, tex_guidance_strength, tex_guidance_rescale, tex_rescale_t,
                decimation_target, texture_size,
                req, progress,
            )

        generate_btn.click(
            prepare_and_generate,
            inputs=[
                image_input, mode, pipeline_type, randomize_seed, seed,
                ss_steps, ss_guidance_strength, ss_guidance_rescale, ss_rescale_t,
                shape_steps, shape_guidance_strength, shape_guidance_rescale, shape_rescale_t,
                tex_steps, tex_guidance_strength, tex_guidance_rescale, tex_rescale_t,
                decimation_target, texture_size,
            ],
            outputs=[output_info, video_output, model_output, download_btn],
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
