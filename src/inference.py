#!/usr/bin/env python3
"""
Inference Script for DPO-Aligned Model
=====================================
A simple Python script to load the final DPO-aligned model and answer user questions.

Usage:
    python inference.py --model_path <path_to_model> [--device cuda]
    
    Or import as a module:
    from inference import ModelInference
    inference = ModelInference(model_path)
    answer = inference.generate_answer("Your question here")
"""

import os
import sys
import argparse
import warnings
import torch
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings("ignore")

# Default configuration
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelInference:
    """
    DPO-Aligned Model Inference Engine
    Handles model loading, tokenizer setup, and response generation.
    """
    
    def __init__(
        self,
        model_path: str,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        device: str = DEFAULT_DEVICE,
        load_in_4bit: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the DPO-aligned model
            max_seq_length: Maximum sequence length for the model
            device: Device to load model on ('cuda' or 'cpu')
            load_in_4bit: Whether to use 4-bit quantization
            verbose: Whether to print status messages
        """
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        
        self._log("Loading model and tokenizer...")
        self._load_model()
        
    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def _load_model(self) -> None:
        """Load model and tokenizer from specified path."""
        try:
            # Import here to avoid requiring unsloth/transformers if not using inference
            from unsloth import FastLanguageModel
            from transformers import AutoTokenizer
            
            # Validate model path
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            # Load model
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=self.load_in_4bit,
                device_map=self.device,
            )
            
            # Setup tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            self._log(f"✓ Model loaded from {self.model_path}")
            self._log(f"✓ Device: {self.device}")
            
            if torch.cuda.is_available():
                self._log(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            
        except ImportError as e:
            raise ImportError(
                "Required libraries not installed. Please install unsloth and transformers:\n"
                "  pip install unsloth transformers torch\n"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def generate_answer(
        self,
        question: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> str:
        """
        Generate an answer to a user question.
        
        Args:
            question: User question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0+ = more random)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated answer string
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Prepare prompt with instruction template
            prompt = self._format_prompt(question)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length
            ).to(self.device)
            
            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}") from e
    
    @staticmethod
    def _format_prompt(question: str) -> str:
        """
        Format question using instruction template.
        
        Args:
            question: User question
            
        Returns:
            Formatted prompt string
        """
        question = str(question).strip()
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"
        return prompt
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def interactive_chat(inference_engine: ModelInference) -> None:
    """
    Interactive chatbot mode for continuous questioning.
    
    Args:
        inference_engine: Initialized ModelInference instance
    """
    print("\n" + "=" * 70)
    print("INTERACTIVE Q&A MODE")
    print("=" * 70)
    print("Ask questions about reimbursement and policies.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        try:
            question = input("\n[Question] ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n✓ Thank you for using the Q&A system. Goodbye!")
                break
            
            if not question:
                print("⚠ Please enter a question.")
                continue
            
            print("\n[Generating response...]")
            answer = inference_engine.generate_answer(question)
            print(f"\n[Answer]\n{answer}")
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\n✓ Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            continue


def main():
    """Main function - handle command line arguments and run inference."""
    parser = argparse.ArgumentParser(
        description="DPO-Aligned Model Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default model path
  python inference.py --model_path ./outputs/stage3_dpo_final_merged
  
  # Single question
  python inference.py --model_path ./outputs/stage3_dpo_final_merged --question "How do I apply for reimbursement?"
  
  # CPU inference
  python inference.py --model_path ./outputs/stage3_dpo_final_merged --device cpu
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the DPO-aligned model directory"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (if not provided, enter interactive mode)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (0.0 = deterministic)"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling parameter"
    )
    
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization (use full precision)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress info messages"
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference = ModelInference(
            model_path=args.model_path,
            device=args.device,
            load_in_4bit=not args.no_4bit,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"✗ Failed to initialize inference: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run inference
    try:
        if args.question:
            # Single question mode
            print(f"\n[Question] {args.question}")
            print("\n[Generating response...]")
            answer = inference.generate_answer(
                args.question,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"\n[Answer]\n{answer}\n")
        else:
            # Interactive mode
            interactive_chat(inference)
    except Exception as e:
        print(f"\n✗ Error during inference: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        inference.clear_memory()


if __name__ == "__main__":
    main()
