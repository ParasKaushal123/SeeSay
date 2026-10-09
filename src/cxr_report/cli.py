"""Command-line interface for CXR report generation."""

import argparse
import json
import sys
import platform
from pathlib import Path

from .model import load_medgemma
from .generator import generate_report
from .output import format_result, save_result
from .prompts import chest_xray_prompt
from .images import image_sha256
from .batch import load_manifest, run_batch

def print_info():
    """Print runtime information."""
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "Not installed"
        
    for k, v in info.items():
        print(f"{k}: {v}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="CXR Report Generation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Generate subcommand
    generate_parser = subparsers.add_parser("generate", help="Generate report for a single image")
    generate_parser.add_argument("--image", required=True, help="Path to the input chest X-ray image")
    generate_parser.add_argument("--output", help="Path to save output JSON (default: stdout)")
    generate_parser.add_argument("--indication", help="Clinical indication")
    generate_parser.add_argument("--checkpoint", default="google/medgemma-1.5-4b-it", help="Model checkpoint")
    generate_parser.add_argument("--quantize-4bit", action="store_true", help="Enable 4-bit quantization")
    generate_parser.add_argument("--max-new-tokens", type=int, default=384, help="Max new tokens to generate")
    
    # Batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Batch process multiple images from a manifest")
    batch_parser.add_argument("--manifest", required=True, help="Path to input manifest JSONL")
    batch_parser.add_argument("--output", required=True, help="Path to output JSONL")
    batch_parser.add_argument("--checkpoint", default="google/medgemma-1.5-4b-it", help="Model checkpoint")
    batch_parser.add_argument("--quantize-4bit", action="store_true", help="Enable 4-bit quantization")
    batch_parser.add_argument("--max-new-tokens", type=int, default=384, help="Max new tokens to generate")
    batch_parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable skipping already completed studies")
    
    # Info subcommand
    subparsers.add_parser("info", help="Print runtime info")
    
    args = parser.parse_args()
    
    if args.command == "info":
        print_info()
        return
        
    try:
        # Load model ONCE
        model, processor, model_info = load_medgemma(
            checkpoint=args.checkpoint,
            device_map="auto",
            quantize_4bit=args.quantize_4bit
        )
        
        if args.command == "generate":
            prompt_config = chest_xray_prompt(args.indication)
            result = generate_report(
                model=model,
                processor=processor,
                image_path=args.image,
                indication=args.indication,
                max_new_tokens=args.max_new_tokens,
                prompt_config=prompt_config,
            )
            img_hash = image_sha256(args.image)
            formatted = format_result(
                result=result,
                study_id=Path(args.image).stem,
                image_path=args.image,
                image_hash=img_hash,
                model_info=model_info,
            )
            if args.output:
                save_result(formatted, args.output)
            else:
                print(json.dumps(formatted, indent=2))
                
        elif args.command == "batch":
            manifest = load_manifest(args.manifest)
            stats = run_batch(
                model=model,
                processor=processor,
                model_info=model_info,
                manifest=manifest,
                output_path=args.output,
                resume=args.resume,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"Batch completed: {stats}")
            
    except Exception as e:
        parser.error(str(e))
