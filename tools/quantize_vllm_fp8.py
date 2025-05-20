import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.transformers import (
    QuantizationModifier,
    oneshot,
)

def main():
    parser = argparse.ArgumentParser(description="Quantize a Hugging Face model to FP8 using llm-compressor.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the original Hugging Face model directory or model name (e.g., 'meta-llama/Llama-2-7b-hf')."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory path to save the FP8 quantized model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run quantization on (e.g., 'cuda', 'cuda:0', 'cpu'). Default is 'cuda'."
    )

    args = parser.parse_args()

    print(f"Starting FP8 quantization for model: {args.base_model_path}")
    print(f"Output will be saved to: {args.output_path}")
    print(f"Using device: {args.device}")

    # Ensure device is valid for torch
    if not (args.device == "cpu" or (args.device.startswith("cuda") and torch.cuda.is_available())):
        print(f"Error: Device '{args.device}' is not available. Ensure CUDA is available if specifying a CUDA device, or use 'cpu'.")
        return

    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        print("Tokenizer loaded.")

        # Load model
        print(f"Loading model from {args.base_model_path} to device {args.device}...")
        # Using device_map should handle placing the model on the specified device.
        # For multi-GPU, device_map="auto" or a custom map would be needed, 
        # but for single device, device_map=args.device should work.
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            device_map=args.device, # Recommended for single GPU by Hugging Face docs
            torch_dtype="auto"      # Use "auto" or specific torch.float16/torch.bfloat16 if needed
        )
        print("Model loaded.")

        # Define the quantization recipe
        # Targets "Linear" for all linear layers.
        # Scheme "FP8_DYNAMIC" uses dynamic E4M3 format for FP8.
        # Ignores "lm_head" as it's typically not quantized or quantized differently.
        print("Defining quantization recipe...")
        recipe = QuantizationModifier(
            targets="Linear", 
            scheme="FP8_DYNAMIC", 
            ignore=["lm_head"] 
        )
        print("Recipe defined.")

        # Apply one-shot quantization
        # The `dev` argument in `oneshot` specifies the device for quantization operations.
        print(f"Applying one-shot quantization on device: {args.device}...")
        oneshot(model=model, recipe=recipe, dev=args.device)
        print("One-shot quantization applied.")

        # Create the output directory if it doesn't exist
        print(f"Creating output directory: {args.output_path} if it doesn't exist...")
        os.makedirs(args.output_path, exist_ok=True)
        print("Output directory ensured.")

        # Save the quantized model
        print(f"Saving quantized model to: {args.output_path}...")
        model.save_pretrained(args.output_path)
        print("Quantized model saved.")

        # Save the tokenizer
        print(f"Saving tokenizer to: {args.output_path}...")
        tokenizer.save_pretrained(args.output_path)
        print("Tokenizer saved.")

        print("\nFP8 quantization process completed successfully!")
        print(f"Quantized model and tokenizer are saved in: {args.output_path}")
        print("Note: Ensure 'llm-compressor' and necessary CUDA drivers (for GPU quantization) are installed in the environment where you run this script.")

    except ImportError:
        print("Error: llm-compressor or transformers library not found. Please install them.")
        print("You can typically install them using: pip install llm-compressor[torch,transformers]")
    except Exception as e:
        print(f"An error occurred during the quantization process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
