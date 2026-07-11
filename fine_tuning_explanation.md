# Fine-Tuning Explanation: Preference-Based Alignment (DPO)

## Overview

This document provides a comprehensive explanation of the three-stage fine-tuning pipeline that combines Supervised Fine-Tuning (SFT) with Direct Preference Optimization (DPO) to create a domain-specialized, instruction-following, and preference-aligned language model for medical imaging systems.

---

## Three-Stage Pipeline Architecture

### Complete Training Flow

```
BASE MODEL (TinyLLaMA 1.1B parameters)
    ↓
[STAGE 1: Domain Adaptation]
    ├─ Input: Raw medical equipment PDF text
    ├─ Method: Causal Language Modeling (CLM)
    ├─ Purpose: Learn domain-specific vocabulary and concepts
    ├─ Output: Domain-aware model
    └─ Learning Rate: 2e-4
    ↓
[STAGE 2: Instruction Fine-Tuning]
    ├─ Input: Stage 1 model + Instruction-Response pairs
    ├─ Method: Supervised Fine-Tuning (SFT)
    ├─ Purpose: Teach structured instruction following
    ├─ Output: Instruction-following model
    └─ Learning Rate: 1e-4
    ↓
[STAGE 3: Preference Alignment]
    ├─ Input: Stage 2 model + Preference pairs (chosen/rejected)
    ├─ Method: Direct Preference Optimization (DPO)
    ├─ Purpose: Align model with human preferences
    ├─ Output: Preference-aligned, instruction-following model
    └─ Learning Rate: 5e-5
    ↓
FINAL MODEL (Production-Ready)
```

---

## Stage 1: Non-Instruction Domain Adaptation

### Purpose
Expose the base model to domain-specific language and knowledge through raw, unstructured text from medical equipment documentation.

### Data Processing

**Input**: Medical equipment PDF documents
- CT scanner specifications and troubleshooting
- MRI system components and maintenance
- Ultrasound equipment guides
- Component failure scenarios

**Processing Pipeline**:
1. **Extraction**: Extract text from each PDF page using PyMuPDF
2. **Cleaning**: Normalize Unicode, remove artifacts, fix word breaks
3. **Chunking**: Split into 512-character overlapping chunks (50-char overlap)
4. **Filtering**: Keep chunks ≥100 characters minimum
5. **Dataset Creation**: Create HuggingFace Dataset

**Example Processing**:
```
Raw Text: "The X-ray tube generates radiation that passes through 
          the patient while detectors convert transmitted photons 
          into electrical signals. Common failures include tube arcing, 
          tube overheating..."
          
Cleaned: "The X-ray tube generates radiation that passes through the 
         patient while detectors convert transmitted photons into 
         electrical signals. Common failures include tube arcing, tube 
         overheating..."
         
Chunks: [
  "The X-ray tube generates radiation... [truncated to 512 chars]",
  "...electrical signals. Common failures include... [next chunk]"
]
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 2e-4 | High LR for rapid domain adaptation |
| **Batch Size** | 2 | Memory efficiency with 4-bit quantization |
| **Gradient Accumulation** | 4 | Effective batch size = 8 |
| **Epochs** | 3 | Sufficient exposure to domain content |
| **Warmup Steps** | 50 | Gradual LR increase for stability |
| **Max Seq Length** | 2048 | Captures full context |
| **Packing** | Enabled | Reduces padding overhead |
| **Quantization** | 4-bit (QLoRA) | 4x memory reduction |

### LoRA Configuration

```python
LORA_R = 16              # Rank: 16-dimensional adaptation space
LORA_ALPHA = 32          # Scaling: 2x amplification of LoRA output
LORA_DROPOUT = 0.05      # Regularization: 5% dropout
TARGET_MODULES = [
    "q_proj",            # Query projection (attention)
    "k_proj",            # Key projection (attention)
    "v_proj",            # Value projection (attention)
    "o_proj",            # Output projection (attention)
]
```

### Key Learning Outcomes

After Stage 1, the model learns:
- ✓ Medical equipment terminology (gantry, detector, tube, etc.)
- ✓ Component relationships and interactions
- ✓ Common failure modes and symptoms
- ✓ Maintenance procedures and schedules
- ✓ Technical diagnostic language

**Memory Footprint**:
- Base model (4-bit): ~2GB VRAM
- LoRA adapter: ~50MB storage
- Total training VRAM: ~3GB

---

## Stage 2: Instruction-Based Fine-Tuning

### Purpose
Fine-tune the domain-aware model from Stage 1 to follow structured instructions and generate professional, well-formatted responses.

### Data Format: Alpaca-Style Instructions

**Input Format (JSONL)**:
```json
{
  "instruction": "What causes CT Scanner failure scenario?",
  "input": "",
  "output": "Common causes include power issues, cooling problems, cable damage, component aging, calibration drift, communication faults, or mechanical wear. Diagnosis should begin with visual inspection and system logs."
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

**Generated Training Text**:
```
### Instruction:
What causes CT Scanner failure scenario?

### Response:
Common causes include power issues, cooling problems, cable damage, component aging, 
calibration drift, communication faults, or mechanical wear. Diagnosis should begin 
with visual inspection and system logs.
```

### Sample Training Data

| # | Instruction | Output Summary |
|---|-----------|-----------------|
| 1 | What causes CT Scanner failure? | Power, cooling, component aging, calibration drift, communication faults |
| 2 | What preventive maintenance for Ultrasound? | Routine cleaning, connector inspection, calibration, cooling check, log review |
| 3 | PET-CT component function? | Performs designated operational role; inspect per service manual if abnormal |
| 4 | Mammography component function? | Performs designated operational role; inspect per service manual if abnormal |
| 5 | MRI image quality degradation? | Calibration errors, detector issues, motion artifacts, electronic noise, reconstruction problems |

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Starting Model** | Stage 1 merged | Inherits domain knowledge |
| **Learning Rate** | 1e-4 | Lower than Stage 1, protects domain knowledge |
| **Batch Size** | 2 | Consistent with Stage 1 |
| **Epochs** | 3 | Sufficient instruction examples processing |
| **Warmup Steps** | 50 | Stable learning rate schedule |
| **Optimizer** | AdamW 8-bit | Memory-efficient gradient updates |
| **Precision** | bfloat16/fp16 | Auto-detected based on hardware |

### Key Learning Outcomes

After Stage 2, the model learns to:
- ✓ Follow structured instruction formats
- ✓ Generate complete, well-formed answers
- ✓ Organize responses logically
- ✓ Maintain professional tone
- ✓ Provide actionable diagnostic steps

**Example Output Improvement**:

Base Model:
```
"Check power supply and restart the system. Contact manufacturer if problems persist."
```

Stage 2 Model:
```
"Ring artifacts typically indicate detector calibration issues or failed detector channels. 
First, verify detector calibration status in system software. Perform diagnostic mode 
inspection of detectors. Review system logs for detector error codes. Execute full 
detector recalibration if necessary. Verify normal operation post-calibration."
```

---

## Stage 3: Direct Preference Optimization (DPO)

### Purpose
Align the instruction-following model with human preferences by training on preference pairs (chosen vs. rejected responses) without requiring reward model training.

### What is DPO?

**Traditional RLHF vs DPO**:

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward Model** | Required (separate training) | Not required |
| **Complexity** | High (3+ training stages) | Lower (integrated into SFT) |
| **Stability** | Can be unstable | More stable |
| **Training Data** | Scores/ratings + trajectories | Preference pairs (chosen/rejected) |
| **Efficiency** | High GPU memory requirement | Lower memory (comparable to SFT) |

**DPO Formula** (simplified):
```
Loss = -log(σ(β(log p(chosen) - log p(rejected))))
       where σ is sigmoid function, β controls preference strength
```

**Interpretation**:
- Maximize likelihood of "chosen" responses
- Minimize likelihood of "rejected" responses
- β=0.1 means preferences have moderate weight (not too extreme)

### Preference Dataset Format

**JSONL Format**:
```json
{
  "prompt": "Why is image quality degraded on PET-CT case 1?",
  "chosen": "Image degradation may result from calibration errors, detector or probe issues, motion artifacts, electronic noise, or reconstruction problems. Recommended workflow: confirm the symptom, inspect related components, review alarms, perform diagnostics and calibration, isolate the root cause, replace faulty parts if necessary, and verify normal operation.",
  "rejected": "Restart the equipment and ignore any alarms because they are usually not important."
}
```

**Key Characteristics**:
- `prompt`: User query or instruction
- `chosen`: Preferred response (high-quality, correct, professional)
- `rejected`: Suboptimal response (incorrect, unsafe, unprofessional)

### Sample Preference Data

| Prompt | Chosen (High-Quality) | Rejected (Low-Quality) |
|--------|----------------------|----------------------|
| "Why is image quality degraded?" | Systematic diagnosis: calibration → detectors → artifacts → electronic noise | "Restart and ignore alarms" |
| "What preventive maintenance?" | Routine procedures with schedule details | Generic unhelpful response |
| "How to troubleshoot probe?" | Step-by-step diagnostic procedure | Vague or unsafe instructions |

### DPO Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 5e-5 | Lowest of all stages (fine refinement) |
| **Epochs** | 2 | Preference signal is strong, fewer epochs needed |
| **Batch Size** | 2 | Consistent across all stages |
| **Beta (DPO Strength)** | 0.1 | Moderate preference weighting |
| **Loss Type** | Sigmoid | Standard DPO loss |
| **Max Seq Length** | 2048 | Same as previous stages |
| **Optimizer** | AdamW 8-bit | Memory efficiency |

**Beta Parameter Explanation**:
- β → 0: Minimal preference weighting (almost no alignment)
- β = 0.1: **Balanced** (typical, used here)
- β = 1.0: Strong preference weighting
- β → ∞: Extreme preference enforcement

### DPO Training Process

```python
# 1. Load Stage 2 SFT model
sft_model = load_pretrained(STAGE2_MERGED_DIR)

# 2. Initialize DPO Trainer (creates reference model automatically)
dpo_trainer = DPOTrainer(
    model=sft_model,
    ref_model=None,  # Auto-creates from sft_model
    train_dataset=preference_pairs,
    args=dpo_config,
)

# 3. Training loop
# For each preference pair:
#   - Forward pass on chosen response
#   - Forward pass on rejected response
#   - Compute preference loss
#   - Update model weights
#   - Keep reference model fixed (copied periodically)

dpo_trainer.train()

# 4. Save aligned model
dpo_trainer.save_model(output_dir)
```

### Expected Improvements from DPO

**Before DPO (Stage 2 SFT)**:
- Follows instructions well
- May still generate plausible but incorrect information
- Tone might be inconsistent

**After DPO (Stage 3)**:
- More accurate and helpful responses
- Consistent professional tone
- Prioritizes safety and correctness
- Avoids contradicting training preferences
- Better structured diagnostics

**Example Comparison**:

**Query**: "The CT gantry will not rotate after startup. What should I check?"

**Stage 2 Response**:
```
Check the power supply and connections. Verify the gantry motor is functioning. 
Look for loose wires or damaged components. Contact service if problem persists.
```

**Stage 3 Response (After DPO)**:
```
First, verify power and controller connection status through system diagnostics. 
Check gantry motor encoder signals and rotation feedback sensors. Review system 
logs for motor fault codes. Inspect mechanical components for physical obstructions 
or bearing wear. Perform gantry calibration routine. If persistent, test motor 
continuity and consider replacement. Contact service with fault codes if self-diagnosis 
insufficient.
```

---

## Key Differences Between Stages

| Aspect | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| **Input Data** | Raw PDF text | Q&A pairs | Preference pairs |
| **Training Method** | CLM (next token) | SFT (instruction) | DPO (preferences) |
| **Learning Rate** | 2e-4 | 1e-4 | 5e-5 |
| **Model Input** | Base model | Stage 1 merged | Stage 2 merged |
| **Primary Goal** | Domain knowledge | Format & structure | Quality & safety |
| **Output Quality** | Domain-aware | Professional responses | Preferred responses |
| **Training Stability** | High | High | High (no reward model) |
| **Data Requirements** | Large corpus | Moderate Q&A set | Small preference pairs |

---

## Memory and Performance Optimization

### Quantization Strategy

```
Full Precision (fp32): 4B params × 4 bytes = 16GB
    ↓
Half Precision (fp16): 4B params × 2 bytes = 8GB
    ↓
4-bit Quantization (QLoRA): 4B params ÷ 4 = 4GB
    + LoRA adapters (50MB) = 4.05GB
```

### Gradient Checkpointing

**Without Checkpointing**:
- Forward pass: store all activations (high memory)
- Backward pass: use stored activations (fast)

**With Checkpointing** (enabled):
- Forward pass: recompute selectively (lower memory)
- Backward pass: recompute needed activations (slower but feasible)
- Trade-off: ~20% slower, 40% less memory

### Packing Strategy

**Without Packing**:
```
[SHORT_TEXT_1][PAD][PAD][PAD]...  (512 tokens)
[SHORT_TEXT_2][PAD][PAD][PAD]...  (512 tokens)
Waste: 60% padding
```

**With Packing** (enabled):
```
[TEXT_1][TEXT_2][TEXT_3][TEXT_4]  (512 tokens)
No padding waste, improved efficiency
```

---

## Inference and Generation

### Generation Parameters

```python
generation_config = {
    "max_new_tokens": 256,      # Maximum output length
    "do_sample": True,           # Non-greedy sampling
    "temperature": 0.7,          # Randomness (0=deterministic, 1=very random)
    "top_p": 0.9,               # Nucleus sampling (keep top 90% probability)
    "repetition_penalty": 1.1,   # Penalize token repetition
}
```

### Inference Pipeline

```python
# 1. Prepare prompt
prompt = f"""### Instruction:
{user_instruction}

### Response:
"""

# 2. Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 3. Generate
with torch.inference_mode():
    outputs = model.generate(**inputs, **generation_config)

# 4. Decode
response = tokenizer.decode(outputs[0][inputs.shape[-1]:])
```

---

## Model Artifacts and Deployment

### Saved Models

```
outputs/
├── stage1_non_instruction_adapter/    # LoRA weights (~50MB)
├── stage1_non_instruction_merged/     # Standalone model (~15GB)
├── stage2_instruction_adapter/        # LoRA weights (~50MB)
├── stage2_instruction_merged/         # Standalone model (~15GB)
├── stage3_dpo_adapter/                # LoRA weights (~50MB)
├── stage3_dpo_merged/                 # Standalone model (~15GB) ← FINAL
└── logs/
    ├── stage1_logs/
    ├── stage2_logs/
    └── stage3_logs/
```

### Loading Final Model

```python
# Load final preference-aligned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="stage3_dpo_final_merged",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Set up for inference
FastLanguageModel.for_inference(model)

# Generate response
response = model.generate(
    inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)
```

---

## Advantages of Three-Stage Approach

### 1. **Progressive Complexity**
   - Stage 1: Raw knowledge → Stage 2: Format → Stage 3: Preference
   - Each stage builds on previous one

### 2. **Reduced Catastrophic Forgetting**
   - Lower learning rates in later stages protect earlier learning
   - Stage 3 (LR=5e-5) most conservative

### 3. **Modular Architecture**
   - Can stop at any stage
   - Each stage produces usable model
   - Adapters can be composed

### 4. **Training Efficiency**
   - No separate reward model needed (vs traditional RLHF)
   - Gradient checkpointing saves memory
   - QLoRA enables training on consumer GPUs

### 5. **Preference Alignment Without RLHF**
   - DPO simpler than RLHF
   - More stable training
   - Comparable or better results

---

## Evaluation Metrics

### Stage-by-Stage Improvements

```
Base Model (Generic)
    ↓
+ Stage 1: Domain Knowledge
    Domain Accuracy: ↑↑↑, Relevance: ↑↑↑
    
+ Stage 2: Instruction Following
    Response Quality: ↑↑, Professionalism: ↑↑↑
    
+ Stage 3: Preference Alignment
    Safety: ↑↑, Helpfulness: ↑, Tone: ↑↑

Final Model: ✓ Domain + Professional + Preferred
```

### Qualitative Assessment

| Dimension | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| **Domain Knowledge** | High | High | High |
| **Instruction Following** | Low | High | High |
| **Response Structure** | Uneven | Professional | Professional |
| **Safety** | Medium | Medium | High |
| **Preference Alignment** | None | None | High |
| **Overall Quality** | Good | Very Good | Excellent |

---

## Best Practices

1. **Progressive Learning Rates**: Start high (2e-4), decrease gradually
2. **Seed Management**: Set SEED=42 for reproducibility across all stages
3. **Memory Management**: Call `clear_gpu_memory()` between stages
4. **Data Quality**: Use high-quality preference pairs for DPO
5. **Testing**: Evaluate at each stage to monitor improvements
6. **Validation**: Reserve preference pairs for evaluation

---

## Common Pitfalls and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Diverging Loss** | LR too high | Reduce LR or use warmup |
| **Model Collapse** | Too aggressive DPO (high β) | Reduce β to 0.05-0.1 |
| **Slow Training** | Inefficient data loading | Enable packing, use gradient accumulation |
| **Poor Alignment** | Low-quality preference data | Validate training data quality |
| **OOM Errors** | Batch size too large | Reduce batch size, enable checkpointing |

---

## Conclusion

The three-stage pipeline (SFT + DPO) provides a comprehensive approach to fine-tuning language models for specialized domains while maintaining high-quality, user-preferred outputs. By combining domain adaptation, instruction fine-tuning, and preference optimization, the final model delivers domain expertise, structural professionalism, and human-aligned responses.

**Next Steps**:
1. Deploy Stage 3 model for production inference
2. Monitor response quality in real-world usage
3. Collect user preferences for iterative improvement
4. Fine-tune subsequent models with new preference data
