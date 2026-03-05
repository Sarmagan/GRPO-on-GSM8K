# GSM8K Mathematical Reasoning with GRPO Fine-tuning

Fine-tuning a 3B parameter language model to solve grade school math problems using Group Relative Policy Optimization (GRPO)

---

## Model

| Property | Value |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |
| Parameters | ~3B |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| Adapter | LoRA (rank 32, alpha 64) |
| Trainable params | ~1-2% of total |
| Training steps | 1,000 |

**LoRA target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## Dataset

**GSM8K** (Grade School Math 8K) by OpenAI — a dataset of 8,500 linguistically diverse grade school math word problems requiring multi-step reasoning.

| Split | Examples |
|---|---|
| Train | 7,473 |
| Test | 1,319 |

---

## Training Method: GRPO

GRPO (Group Relative Policy Optimization) is a reinforcement learning algorithm that optimizes a language model using reward functions instead of gold-label completions. Rather than telling the model *what* to output, reward functions score its outputs and it learns to maximize those scores.

### Reward Functions

Four complementary reward functions were used:

| Reward Function | Max Score | Description |
|---|---|---|
| `match_format_exactly` | 3.0 | Full score for perfect adherence to the structured output format |
| `match_format_approximately` | 2.0 | Partial credit (+0.5 per correct tag) for partially correct formatting |
| `check_answer_correctness` | 3.0 | Graduated scoring: exact match (3.0), within 10% (1.5), within 20% (0.5), wrong (-0.5) |
| `check_numbers_extraction` | 1.5 | Binary score for correctly extracting a numerical answer from the solution section |

### Output Format

The model was trained to produce structured responses using custom delimiter tokens:

```
<start_working_out>
Step-by-step reasoning...
<end_working_out>
<SOLUTION>
42
</SOLUTION>
```

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Learning rate | 5e-6 |
| Batch size (per device) | 8 |
| Gradient accumulation steps | 2 |
| Effective batch size | 16 |
| Max completion length | 1,024 tokens |
| Max grad norm | 0.1 |
| Gradient checkpointing | ✅ |
| Optimizer | AdamW (TRL default) |
| Decoding (eval) | Greedy (temperature=0) |

---

## Results

Evaluated on the full GSM8K test set (1,319 examples) after 1,000 training steps.

| Metric | Base Model | GRPO (1k steps) | Delta |
|---|---|---|---|
| **Accuracy** | 64.90% (856/1319) | **71.87% (948/1319)** | **+6.97%** |
| Format compliance | 7.28% (96/1319) | 53.60% (707/1319) | +46.32% |
| No answer parsed | 1.59% (21/1319) | 0.53% (7/1319) | -1.06% |

### Key Takeaways

- **Accuracy improved by +6.97%** after only 1,000 training steps, demonstrating GRPO's sample efficiency on mathematical reasoning.
- **Format compliance increased by ~7x** (7.28% → 53.60%), confirming that reward shaping effectively teaches structured output behaviour.
- The model achieves ~71.87% accuracy while only producing compliant formatted responses 53.60% of the time, meaning it still answers correctly in many non-formatted responses — residual capability from the base model's pre-existing math skills.
- Further training steps are expected to continue improving both format compliance and accuracy in tandem.

---
