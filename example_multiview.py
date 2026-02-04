"""
Multi-view 3D Generation Example for TRELLIS.2

This example demonstrates using multiple input images (views) to generate
higher-quality 3D assets. The multi-view approach aggregates predictions
from different viewpoints during the denoising process.

Usage:
    python example_multiview.py --front path/to/front.png --back path/to/back.png
    python example_multiview.py --images img1.png img2.png img3.png --mode multidiffusion
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2MultiViewPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel


def main():
    parser = argparse.ArgumentParser(description='TRELLIS.2 Multi-View 3D Generation')
    parser.add_argument('--front', type=str, help='Path to front view image')
    parser.add_argument('--back', type=str, help='Path to back view image')
    parser.add_argument('--images', nargs='+', type=str, help='Paths to multiple view images')
    parser.add_argument('--mode', type=str, default='stochastic',
                        choices=['stochastic', 'multidiffusion'],
                        help='Aggregation mode (default: stochastic)')
    parser.add_argument('--output', type=str, default='sample_multiview',
                        help='Output filename prefix (default: sample_multiview)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pipeline-type', type=str, default='1024_cascade',
                        choices=['512', '1024', '1024_cascade', '1536_cascade'],
                        help='Pipeline type (default: 1024_cascade)')
    parser.add_argument('--no-video', action='store_true', help='Skip video rendering')
    args = parser.parse_args()

    # Collect input images
    if args.images:
        image_paths = args.images
    elif args.front and args.back:
        image_paths = [args.front, args.back]
    else:
        # Default example: use the T.png image twice (for testing)
        print("No images specified. Using default example image.")
        image_paths = ['assets/example_image/T.png', 'assets/example_image/T.png']

    print(f"Loading {len(image_paths)} images:")
    for p in image_paths:
        print(f"  - {p}")

    images = [Image.open(p) for p in image_paths]

    # Setup Environment Map for rendering
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))

    # Load Pipeline
    print("\nLoading TRELLIS.2 Multi-View Pipeline...")
    pipeline = Trellis2MultiViewPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    # Run Multi-View Generation
    print(f"\nGenerating 3D mesh with {len(images)} views using '{args.mode}' mode...")
    print(f"Pipeline type: {args.pipeline_type}")

    mesh = pipeline.run_multi_image(
        images,
        seed=args.seed,
        mode=args.mode,
        pipeline_type=args.pipeline_type,
    )[0]

    # Simplify mesh (nvdiffrast has limits)
    mesh.simplify(16777216)

    # Render Video
    if not args.no_video:
        print(f"\nRendering video to {args.output}.mp4...")
        video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
        imageio.mimsave(f"{args.output}.mp4", video, fps=15)

    # Export to GLB
    print(f"\nExporting to {args.output}.glb...")
    glb = o_voxel.postprocess.to_glb(
        vertices            =   mesh.vertices,
        faces               =   mesh.faces,
        attr_volume         =   mesh.attrs,
        coords              =   mesh.coords,
        attr_layout         =   mesh.layout,
        voxel_size          =   mesh.voxel_size,
        aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target   =   1000000,
        texture_size        =   4096,
        remesh              =   True,
        remesh_band         =   1,
        remesh_project      =   0,
        verbose             =   True
    )
    glb.export(f"{args.output}.glb", extension_webp=True)

    print(f"\nDone! Output files:")
    if not args.no_video:
        print(f"  - {args.output}.mp4 (preview video)")
    print(f"  - {args.output}.glb (3D mesh)")


if __name__ == '__main__':
    main()
