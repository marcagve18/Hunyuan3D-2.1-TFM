import os
import argparse
from pathlib import Path
import gc

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen-Image-Edit-2511")
    p.add_argument("--input_dir", required=True, help="Directory containing albedo_XX.png files")
    p.add_argument("--output_dir", required=True, help="Where to write enhanced images (same filenames)")
    p.add_argument("--prompt_image", required=True, help="Path to the original image prompt (context)")
    p.add_argument("--glob", default="albedo_*.png", help="Glob pattern for input images inside input_dir")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--num_inference_steps", type=int, default=24)
    p.add_argument("--true_cfg_scale", type=float, default=4.0)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prompt", default=(
        "Enhance the first image: increase clarity, reduce noise, sharpen fine details, "
        "improve texture fidelity. Preserve geometry and colors. Use the second image only as reference."
    ))
    p.add_argument("--negative_prompt", default=" ")
    p.add_argument("--enable_attention_slicing", action="store_true", default=True,
                   help="Enable attention slicing to reduce memory usage")
    p.add_argument("--enable_vae_slicing", action="store_true", default=True,
                   help="Enable VAE slicing to reduce memory usage")
    p.add_argument("--cpu_offload", action="store_true", default=False,
                   help="Offload model components to CPU when not in use")
    return p.parse_args()


def main():
    args = parse_args()

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU CHECK] Device: {device}")

    # Check GPU memory at subprocess start
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = total - allocated
        print(f"[GPU CHECK] Total: {total:.2f}GB, Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")
        
        if allocated > 15:  # More than 15GB already in use
            print(f"[WARNING] GPU memory heavily used ({allocated:.2f}GB). Clearing cache...")
            torch.cuda.empty_cache()
            gc.collect()
            allocated_after = torch.cuda.memory_allocated() / 1024**3
            print(f"[GPU CHECK] After cleanup: {allocated_after:.2f}GB allocated")


    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt image once and keep in memory
    prompt_img = Image.open(args.prompt_image).convert("RGB")

    # Load pipeline ONCE with memory optimizations
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_id, 
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    
    # Apply memory optimization techniques
    if args.enable_attention_slicing:
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing(slice_size=1)
    
    if args.enable_vae_slicing:
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
    
    if args.cpu_offload:
        # Sequential CPU offloading - moves model components to CPU when not needed
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to(device)
    
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    # Deterministic generator
    gen = torch.Generator(device=device)
    gen.manual_seed(int(args.seed))

    files = sorted(in_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.glob} in {in_dir}")

    for i, fp in enumerate(files):
        # Load image
        view_img = Image.open(fp).convert("RGB")

        # Run inference
        with torch.inference_mode():  # Disable gradient computation
            out = pipe(
                image=[view_img, prompt_img],
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=int(args.num_inference_steps),
                true_cfg_scale=float(args.true_cfg_scale),
                guidance_scale=float(args.guidance_scale),
                generator=gen,
                num_images_per_prompt=1,
            )

        # Save and immediately free memory
        enhanced = out.images[0].convert("RGB")
        enhanced.save(out_dir / fp.name)
        
        # Explicit cleanup
        del view_img, out, enhanced
        
        # Periodic CUDA cache clearing (every 5 images)
        if device == "cuda" and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Final cleanup
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Enhanced {len(files)} images -> {out_dir}")


if __name__ == "__main__":
    main()