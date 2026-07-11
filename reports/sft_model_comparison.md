# SFT Model Comparison: Instruction-Based Fine-Tuning

## Overview

This document details the Supervised Fine-Tuning (SFT) process for instruction-based model fine-tuning using a two-stage approach: non-instruction pre-training followed by instruction fine-tuning.

## Architecture

### Two-Stage Fine-Tuning Pipeline

```
Base Model (TinyLLaMA 4-bit)
    ↓
[Stage 1: Non-Instruction Fine-Tuning]
    ├─ Training Data: Raw PDF text
    ├─ Method: Causal Language Modeling
    ├─ Output: Stage 1 merged model
    └─ LoRA Adapter saved
    ↓
[Stage 2: Instruction Fine-Tuning]
    ├─ Input Model: Stage 1 merged model
    ├─ Training Data: Instruction-Response pairs (Alpaca format)
    ├─ Method: SFT on structured prompts
    ├─ Output: Stage 2 merged model
    └─ LoRA Adapter saved
    ↓
Final Fine-Tuned Model (Ready for Inference)
```

---

## Stage 1: Non-Instruction Fine-Tuning

### Purpose
Adapt the base model to domain-specific content (medical equipment) by training on raw, unstructured PDF text.

### Input Data
- **Source**: PDF documents containing medical equipment documentation
- **Processing**:
  - Extract text from PDF pages
  - Clean and normalize text (Unicode normalization, remove artifacts)
  - Split into overlapping chunks (512 chars, 50 char overlap)
  - Filter chunks by minimum length (100+ characters)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/tinyllama-bnb-4bit` |
| Learning Rate | 2e-4 |
| Batch Size | 2 |
| Gradient Accumulation Steps | 4 |
| Number of Epochs | 3 |
| Max Sequence Length | 2048 |
| Warmup Steps | 50 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Optimization | AdamW 8-bit |
| Precision | bfloat16 or fp16 (auto-detected) |

### Quantization & Efficiency

- **QLoRA**: 4-bit quantization enabled via model loading
- **Gradient Checkpointing**: Used to reduce memory overhead
- **Packing**: Training samples packed to reduce padding

### Output
- **LoRA Adapter**: Saved to `stage1_non_instruction_adapter`
- **Merged Model**: 16-bit merged model saved to `stage1_non_instruction_merged`
- This merged model becomes the starting point for Stage 2

---

## Stage 2: Instruction-Based Fine-Tuning

### Purpose
Fine-tune the Stage 1 model on instruction-response pairs to teach it to follow structured prompts and generate high-quality responses.

### Input Data Format

**Alpaca-Style Format**:
```json
{
  "instruction": "What causes CT Scanner failure scenario?",
  "input": "",
  "output": "Common causes include power issues, cooling problems, cable damage, component aging, calibration drift, communication faults, or mechanical wear."
}
```

**Prompt Template**:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### Sample Instructions
- "What causes CT Scanner failure scenario?"
- "What preventive maintenance is required for Ultrasound item?"
- "What is the function of the PET-CT component?"
- "Why is image quality degraded on MRI case?"

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Starting Model | Stage 1 merged model |
| Learning Rate | 1e-4 |
| Batch Size | 2 |
| Gradient Accumulation Steps | 4 |
| Number of Epochs | 3 |
| Max Sequence Length | 2048 |
| Warmup Steps | 50 |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Training Dataset Field | `text` (formatted prompt) |
| Packing | Enabled |

### Output
- **LoRA Adapter**: Saved to `stage2_instruction_adapter`
- **Merged Model**: 16-bit merged model saved to `stage2_instruction_merged`
- **Final Model**: Production-ready instruction-following model

---

## Model Comparison Evaluation

### Evaluation Methodology

Compare three models on the same test questions:

1. **Base Model**: Unmodified TinyLLaMA (no fine-tuning)
2. **Stage 1 Model**: After non-instruction fine-tuning (domain knowledge)
3. **Stage 2 Model**: After instruction fine-tuning (domain + instruction-following)

### Sample Test Questions

| # | Question |
|---|----------|
| 1 | My CT scanner is producing circular ring artifacts on every scan. What is the most likely cause and what should I inspect first? |
| 2 | The CT scanner stopped scanning because of a tube overheating alarm. What components are likely responsible? |
| 3 | How does quantization reduce model size? |
| 4 | My MRI system displays a low helium pressure alarm during scanning. What could be causing this? |
| 5 | The ultrasound probe is connected, but the system cannot detect it. How should I troubleshoot this issue? |
| 6 | The CT gantry will not rotate after startup. Which components should I inspect? |
| 7 | MRI images contain zipper artifacts. What usually causes this type of artifact? |
| 8 | The ultrasound image is completely black even though the machine powers on. What are the possible causes? |
| 9 | What preventive maintenance should be performed on a CT scanner to reduce unexpected failures? |
| 10 | How can I differentiate between a detector failure and a detector calibration problem in a CT scanner? |
| 11 | My imaging system repeatedly restarts during operation without displaying any error code. What are the most likely causes? |

### Expected Improvements

| Aspect | Base Model | Stage 1 | Stage 2 |
|--------|-----------|---------|---------|
| **Domain Knowledge** | Generic | High (trained on medical PDF) | High (retains domain knowledge) |
| **Instruction Following** | Low | Medium | High (fine-tuned on structured prompts) |
| **Response Quality** | Generic troubleshooting | Domain-specific but unstructured | Structured, professional responses |
| **Relevance** | General tech advice | Medical equipment focused | Medical equipment + proper format |

### Example Comparison

**Question**: "The CT scanner stopped scanning because of a tube overheating alarm. What components are likely responsible?"

**Base Model Answer** (generic):
- Suggests checking power supply and basic cables
- Generic hardware failure diagnostics

**Stage 1 Model Answer** (domain-aware):
- Mentions cooling system components
- References tube-specific parameters and temperature thresholds
- More specific to medical equipment

**Stage 2 Model Answer** (domain + instruction-aware):
- Structured, professional format
- Specific component inspection order
- Includes preventive maintenance tips
- Follows the prompt format precisely

---

## Key Hyperparameters & Design Choices

### Learning Rate Scheduling

- **Stage 1**: 2e-4 (higher for domain adaptation)
- **Stage 2**: 1e-4 (lower for instruction fine-tuning refinement)

**Rationale**: Higher LR for Stage 1 to quickly adapt to new domain. Lower LR for Stage 2 to carefully fine-tune instruction-following without catastrophic forgetting.

### LoRA Configuration

```python
LORA_R = 16              # Rank - balance between parameter efficiency and expressiveness
LORA_ALPHA = 32          # Scaling factor
LORA_DROPOUT = 0.05      # Regularization
TARGET_MODULES = [
    "q_proj",            # Query projection in attention
    "k_proj",            # Key projection in attention
    "v_proj",            # Value projection in attention
    "o_proj",            # Output projection in attention
    "gate_proj",         # Gate in feed-forward
    "up_proj",           # Up-projection in feed-forward
    "down_proj"          # Down-projection in feed-forward
]
```

### Memory Optimization

- **4-bit Quantization**: 4x reduction in model memory
- **Gradient Checkpointing**: Trade compute for memory (saves ~40% memory)
- **Packing**: Reduces padding waste in training data
- **Flash Attention via Unsloth**: Optimized attention computation

---

## Inference

### Generation Parameters

```python
max_new_tokens = 256
do_sample = True
temperature = 0.7        # Moderate randomness
top_p = 0.9             # Nucleus sampling
repetition_penalty = 1.1 # Discourage repetition
```

### Interactive Bot Usage

```python
def interactive_bot(model, tokenizer, stage_name):
    instruction = "What causes CT Scanner failure?"
    response = generate_response(model, tokenizer, instruction)
    print(f"Response: {response}")
```

---

## Model Artifacts

### Saved Outputs

```
outputs/
├── stage1_logs/                          # Training logs
├── stage1_non_instruction_adapter/       # LoRA adapter weights
├── stage1_non_instruction_merged/        # Merged 16-bit model
├── stage2_logs/                          # Training logs
├── stage2_instruction_adapter/           # LoRA adapter weights
└── stage2_instruction_merged/            # Final merged model (production-ready)
```

### Model Loading

**Load LoRA Adapter Only** (smaller, faster):
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="stage1_non_instruction_adapter",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
```

**Load Merged Model** (standalone, no base model required):
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="stage1_non_instruction_merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
```

---

## Performance Metrics

### Training Metrics

- **Training Time**: Logged per stage
- **Final Loss**: Convergence metric
- **GPU Memory Usage**: Peak allocated and reserved VRAM

### Inference Metrics

- **Generation Speed**: Tokens per second
- **Model Size**: 
  - Base: ~2GB (4-bit)
  - Merged: ~15GB (16-bit, full model)
  - Adapter: ~50MB (LoRA weights only)

---

## Best Practices

1. **Two-Stage Training**: Domain adaptation → instruction fine-tuning prevents catastrophic forgetting
2. **Lower LR in Stage 2**: Protects learned domain knowledge while fine-tuning
3. **Use Merged Models for Production**: Standalone inference without base model dependency
4. **Sample Validation**: Compare outputs on held-out test questions
5. **Memory Management**: Clear GPU cache between stages
6. **Seed Setting**: Reproducibility via fixed SEED=42

---

## Troubleshooting

### GPU Memory Issues
- Reduce batch size or gradient accumulation steps
- Enable gradient checkpointing (already enabled)
- Reduce max_seq_length if not needed

### Low Quality Responses
- Ensure instruction dataset is high quality
- Increase number of epochs
- Reduce learning rate for finer tuning
- Verify Stage 1 model learned domain knowledge

### Model Not Following Instructions
- Verify JSONL format is correct
- Check prompt template matches training format
- Ensure sufficient instruction examples (min 10-50)

---

## References

- **Unsloth**: Efficient fine-tuning library
- **LoRA**: Low-Rank Adaptation (Hu et al., 2021)
- **QLoRA**: Efficient fine-tuning of quantized models (Dettmers et al., 2023)
- **Alpaca Format**: Instruction-following dataset format (Stanford CRFM)
