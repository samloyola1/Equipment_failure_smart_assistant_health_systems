# Base Model Evaluation: Non-Instruction Fine-Tuning

## Overview

This document details the base model evaluation and non-instruction fine-tuning process using domain-specific medical equipment data. The focus is on understanding how a base model adapts to specialized domain knowledge through continuous pre-training on raw text data.

## Base Model Assessment

### Model Selection

| Parameter | Value |
|-----------|-------|
| Base Model | `unsloth/tinyllama-bnb-4bit` |
| Model Size | ~1.1B parameters |
| Architecture | Llama-based transformer |
| Quantization | 4-bit (BNB quantization) |
| Context Window | 4096 tokens (max 512 in this config) |
| Precision | Loaded as bfloat16/fp16 |

### Base Model Characteristics

**Strengths**:
- Lightweight and fast for inference
- General-purpose language understanding
- Pre-trained on diverse internet text (up to training cutoff)
- Good foundation for domain adaptation

**Limitations**:
- No domain-specific medical equipment knowledge
- Generic troubleshooting advice only
- May provide irrelevant responses for specialized questions
- Lacks structured medical terminology

---

## Dataset Preparation: Non-Instruction Fine-Tuning

### Data Source

**Input**: Medical equipment PDF documentation
- Technical specifications
- Troubleshooting guides
- Component descriptions
- Maintenance procedures

### Text Processing Pipeline

```
Raw PDF
    ↓
[PDF Extraction]
    ├─ Extract text per page using PyMuPDF (fitz)
    ├─ Preserve page structure
    └─ Handle multi-page documents
    ↓
[Text Cleaning]
    ├─ Unicode normalization (NFKC)
    ├─ Remove invisible characters (zero-width, BOM)
    ├─ Fix hyphenated word breaks (word-\n → word)
    ├─ Remove page numbers (regex: ^\s*\d+\s*$)
    └─ Normalize whitespace and line breaks
    ↓
[Text Chunking]
    ├─ Split into 512-character chunks
    ├─ 50-character overlap for context preservation
    ├─ Break at sentence boundaries when possible
    └─ Filter chunks < 100 characters
    ↓
[Dataset Creation]
    ├─ Create HuggingFace Dataset objects
    ├─ Include metadata (source_page, chunk_id)
    └─ Format for SFT training
```

### Cleaning Functions

#### Text Normalization
```python
def clean_pdf_text(text: str) -> str:
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    
    # Remove invisible characters
    text = text.replace("\u200b", "").replace("\ufeff", "")
    
    # Fix hyphenated words
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    
    # Remove page numbers
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
    
    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()
```

#### Paragraph Extraction
```python
def extract_paragraphs(text: str) -> List[str]:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\n+", " ", paragraph)
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        
        if paragraph:
            paragraphs.append(paragraph)
    
    return paragraphs
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Pages Extracted** | Depends on PDF size |
| **Paragraphs Generated** | Depends on content |
| **Min Paragraph Length** | 80 characters |
| **Sample Paragraph** | Medical equipment description or troubleshooting step |
| **Total Training Records** | Variable (typically 100-1000 paragraphs) |

### Sample Data Categories

1. **Equipment Descriptions**
   - CT scanner components
   - MRI system architecture
   - Ultrasound probe specifications

2. **Failure Scenarios**
   - Ring artifacts on CT scans
   - Tube overheating alarms
   - Detector calibration issues

3. **Preventive Maintenance**
   - Routine inspection procedures
   - Component replacement schedules
   - Performance monitoring

4. **Troubleshooting Guides**
   - Diagnostic procedures
   - Component testing methods
   - Error code interpretation

---

## Non-Instruction Fine-Tuning (Stage 1)

### Objective

Adapt the base model to the medical equipment domain through causal language modeling on raw, unstructured text. This is "continued pre-training" rather than instruction fine-tuning.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Learning Rate** | 2e-4 |
| **Batch Size** | 1-2 |
| **Gradient Accumulation Steps** | 8 |
| **Warmup Steps** | 5 |
| **Max Steps** | 30 (demo) / 100+ (production) |
| **Max Sequence Length** | 512 |
| **Packing** | Enabled (efficient padding) |
| **Precision** | bfloat16 (if supported) or fp16 |
| **Optimizer** | AdamW 8-bit (memory efficient) |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| **LoRA Rank (r)** | 16 |
| **LoRA Alpha** | 32 |
| **LoRA Dropout** | 0.0 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Bias** | none |
| **Gradient Checkpointing** | unsloth (optimized) |

### Memory Optimization Techniques

1. **4-bit Quantization (QLoRA)**
   - Quantize base model to 4 bits
   - LoRA adapters in full precision
   - ~4x memory reduction

2. **Gradient Checkpointing**
   - Trade compute for memory
   - Recompute activations during backward pass
   - ~40% memory savings

3. **Packing**
   - Pack multiple short sequences into single batch
   - Reduce padding overhead
   - Increase throughput

4. **AdamW 8-bit**
   - Quantize optimizer states to 8 bits
   - Further memory reduction

### Training Metrics

**Training Procedure**:
```python
# Clear GPU cache before training
gc.collect()
torch.cuda.empty_cache()

# Track memory usage
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# Run training
trainer.train()

# Log results
train_time = time.time() - start_time
peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
```

**Recorded Metrics**:
- Training time (seconds)
- Peak allocated VRAM (GB)
- Peak reserved VRAM (GB)
- Final training loss
- Steps completed

---

## Base Model vs. Fine-Tuned Model Comparison

### Evaluation Framework

#### Test Question Categories

| Category | Example |
|----------|---------|
| **Ring Artifacts (CT)** | "My CT scanner is producing circular ring artifacts on every scan. What is the most likely cause?" |
| **Overheating Alarm** | "The CT scanner stopped scanning due to tube overheating. What components are responsible?" |
| **System Architecture** | "What are the main components of a CT scanner?" |
| **MRI Issues** | "My MRI system displays a low helium pressure alarm. What could cause this?" |
| **Probe Issues** | "The ultrasound probe cannot be detected. How to troubleshoot?" |
| **Maintenance** | "What preventive maintenance reduces unexpected equipment failures?" |

### Expected Response Differences

#### Generic Questions (Out-of-Domain)

**Question**: "How does quantization reduce model size?"

**Base Model**:
- General explanation of quantization
- Applies generic deep learning knowledge
- No medical equipment context

**Fine-Tuned Model**:
- Same generic explanation (out-of-domain)
- May not change significantly
- Shows fine-tuned model isn't overtrained

#### Medical Equipment Questions (In-Domain)

**Question**: "The CT scanner stopped scanning because of a tube overheating alarm. What components are likely responsible?"

**Base Model Response**:
- Generic hardware failure suggestions
- "Check power supply and connectors"
- "Try restarting the system"
- Limited technical depth

**Fine-Tuned Model Response**:
- Specific component focus (cooling system, tube power)
- Temperature thresholds and parameters
- Maintenance schedule references
- More professional medical equipment terminology
- Specific diagnostic procedures

### Qualitative Evaluation Criteria

| Criterion | Definition | Base Model | Fine-Tuned |
|-----------|-----------|-----------|-----------|
| **Relevance** | Answer addresses the specific equipment | Low | High |
| **Technical Depth** | Domain-specific terminology used | Low | High |
| **Accuracy** | Information matches training domain | Low | High |
| **Specificity** | Mentions exact components/procedures | Low | High |
| **Completeness** | Covers diagnostic and maintenance aspects | Low | Medium |
| **Professionalism** | Appropriate for technical documentation | Medium | High |

---

## Model Output Analysis

### Ring Artifacts Example

**Base Model Answer**:
> "This could be a problem with the display or a hardware issue with the scanner itself. Try unplugging the device, waiting a few seconds, and plugging it back in. If the problem persists, contact the manufacturer."

**Fine-Tuned Model Answer**:
> "Ring artifacts indicate detector calibration issues or failed detector channels. First, check detector calibration status in the system software. Inspect the detector for damaged channels using diagnostic mode. Verify coolant levels if cooling affects detector performance. Review system logs for detector error codes. If issues persist, perform full detector recalibration or contact service..."

### Analysis

**Base Model**:
- Generic troubleshooting (restart)
- No technical depth
- Suggests manufacturer contact without diagnostics
- Generic solution path

**Fine-Tuned Model**:
- Specific root cause (detector calibration/channels)
- Diagnostic procedures (calibration check, error logs)
- Equipment-specific checks (coolant, detector inspection)
- Maintenance reference integrated naturally
- Professional technical language

---

## Inference Workflow

### Model Loading

```python
def load_unsloth_model_with_lora(model_name_or_path: str):
    """Load base or merged model in 4-bit with fresh LoRA adapter"""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ...],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
    return model, tokenizer
```

### Generation Function

```python
def generate_answer(model, tokenizer, instruction: str, 
                   input_text: str = "", max_new_tokens: int = 150):
    """Generate answer from instruction"""
    
    FastLanguageModel.for_inference(model)
    
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], 
        skip_special_tokens=True
    ).strip()
    
    return response
```

### Generation Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_new_tokens` | 150 | Limit response length |
| `do_sample` | True | Non-greedy sampling |
| `temperature` | 0.7 | Moderate diversity (0=deterministic, 1=random) |
| `top_p` | 0.9 | Nucleus sampling (keep top 90% probability mass) |
| `repetition_penalty` | 1.1 | Discourage token repetition |

---

## Model Artifacts & Storage

### Directory Structure

```
outputs/
├── stage1_non_instruction_adapter/
│   ├── adapter_config.json       # LoRA configuration
│   ├── adapter_model.bin         # LoRA weights (~50-100MB)
│   ├── config.json               # Model config
│   ├── tokenizer.json
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
│
├── stage1_non_instruction_merged/
│   ├── model.safetensors         # Full merged model (~15GB at 16-bit)
│   ├── config.json
│   ├── tokenizer.json
│   ├── special_tokens_map.json
│   └── tokenizer_config.json
│
└── stage1_logs/
    └── events.out.tfevents...    # Training logs
```

### Model Size Comparison

| Format | Size | Use Case |
|--------|------|----------|
| **4-bit Base + LoRA** | ~2GB + 50MB | Memory-efficient fine-tuning |
| **Merged 16-bit** | ~15GB | Standalone inference (no base model) |
| **Quantized Merged** | ~4GB | Efficient inference |

---

## Performance Metrics

### Training Performance

**Stage 1 Training Output Example**:
```
STAGE 1 - NON-INSTRUCTION PDF TRAINING RESULTS
================================================
Train time/sec: 125.45
Peak allocated VRAM/GB: 2.847
Peak reserved VRAM/GB: 3.102
Final training loss: 2.3456
Steps completed: 30
```

### Inference Performance

**Speed Benchmark**:
- Tokens/second: ~50-100 (depends on hardware)
- Time per 150-token response: ~1.5-3 seconds (single A100 GPU)
- Model load time: ~10-15 seconds

**Memory Usage**:
- 4-bit model + LoRA: ~2.5GB VRAM
- Merged 16-bit model: ~15GB VRAM
- CPU RAM: ~500MB (tokenizer + config)

---

## Best Practices for Domain Adaptation

### Data Preparation

1. **Quality Over Quantity**
   - Ensure PDF contains relevant domain knowledge
   - Remove OCR artifacts if using scanned documents
   - Verify text extraction accuracy

2. **Text Cleaning**
   - Always normalize Unicode
   - Remove pagination artifacts
   - Fix broken word hyphenation
   - Standardize whitespace

3. **Chunk Size Selection**
   - 512 chars: Balance between context and variety
   - 50 char overlap: Maintain sentence boundaries
   - Min 80 chars: Filter noise and metadata

### Training Optimization

1. **Learning Rate**
   - Start with 2e-4 for domain adaptation
   - Lower to 1e-4 for refinement stages
   - Monitor loss curve for divergence

2. **Batch Size Strategy**
   - Smaller batches (1-2) for limited VRAM
   - Gradient accumulation compensates
   - Stability vs. efficiency tradeoff

3. **Convergence Monitoring**
   - Watch for training loss plateau
   - Compare base vs. fine-tuned on validation set
   - Early stopping if no improvement (10+ steps)

---

## Troubleshooting Guide

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Out of Memory (OOM)** | Batch size too large | Reduce batch size, enable gradient checkpointing |
| **Slow Training** | Inefficient data loading | Use packing, reduce max_seq_length if possible |
| **Poor Domain Knowledge** | Insufficient training data | Add more domain documents or increase epochs |
| **Generic Responses** | Model not adapting | Check learning rate, verify training data quality |
| **Diverging Loss** | Learning rate too high | Reduce LR (try 1e-4 or 5e-5) |

---

## Conclusion

Non-instruction fine-tuning serves as the critical first stage in domain adaptation. By exposing the base model to medical equipment knowledge through raw text training, the model learns domain-specific language patterns and troubleshooting procedures. This foundation enables the subsequent instruction fine-tuning stage to teach structured response generation while retaining specialized domain knowledge.

**Key Takeaways**:
- Base models lack domain expertise; fine-tuning is essential
- Multi-stage training (non-instruction → instruction) prevents catastrophic forgetting
- Proper data preparation directly impacts model quality
- LoRA + quantization enables efficient GPU-constrained training
- Performance gains are measurable and significant for in-domain questions
