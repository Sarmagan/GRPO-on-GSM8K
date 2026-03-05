import os
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import transformers
import wandb
import trl

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed, GenerationConfig
from peft import LoraConfig, PeftModel, get_peft_model
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from huggingface_hub import login
from typing import Optional, List
from google.colab import userdata
from peft import prepare_model_for_kbit_training

# Select model optimized for instruction-following and reasoning
model_name = "meta-llama/Llama-3.2-3B-Instruct" # 3B parameter model balances capability and memory usage
max_seq_length = 2048                     # Token limit for mathematical problems (reduce if OOM)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit precision (vs 16-bit default)
    bnb_4bit_quant_type="nf4",           # NormalFloat4: optimal for neural network weights
    bnb_4bit_compute_dtype=torch.float16, # Use FP16 for forward/backward passes
    bnb_4bit_use_double_quant=True,      # Further quantize quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,      # Apply 4-bit quantization
    device_map="auto",                   # Auto-distribute across available GPUs/CPU
    trust_remote_code=True,              # Allow custom model code execution
    torch_dtype=torch.float16,           # Use FP16 for non-quantized operations
)

# Load corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True               # Allow custom tokenizer code
)

# Ensure tokenizer has proper padding token for batch processing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded successfully!")
print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Quantized parameters: ~{sum(p.numel() for p in model.parameters() if hasattr(p, 'quant_type')) / 1e6:.1f}M")


lora_config = LoraConfig(
    r=32,                              # Rank: adaptation capacity (16 good for reasoning tasks)
    lora_alpha=64,                     # Scaling factor (typically 2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,                  # Regularization to prevent overfitting
    bias="none",                       # Skip bias adaptation for simplicity
    task_type="CAUSAL_LM",      # Causal language modeling task
)

# Apply LoRA configuration to create trainable adapter
model = get_peft_model(model, lora_config)

# Display parameter efficiency
print("LoRA Training Parameters Summary:")
model.print_trainable_parameters()  # Shows trainable vs total parameters

# Define structured output format for mathematical reasoning
reasoning_start = "<start_working_out>"   # Begin reasoning section
reasoning_end = "<end_working_out>"       # End reasoning section
solution_start = "<SOLUTION>"            # Begin final answer
solution_end = "</SOLUTION>"              # End final answer

# System prompt that teaches the model our desired reasoning structure
system_prompt = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {reasoning_start} and {reasoning_end}
2. Provide your final numerical answer between {solution_start} and {solution_end}
3. Be precise and show all calculation steps clearly.

Example output format:
{reasoning_start}
First, I need to add 2 and 2. 2 + 2 = 4.
{reasoning_end}
{solution_start}
4
{solution_end}"""

print("Format tokens and system prompt defined")
print(f"   Reasoning format: {reasoning_start} ... {reasoning_end}")
print(f"   Solution format: {solution_start} ... {solution_end}")

# Dataset processing utilities
def extract_hash_answer(text):
    """Extract numerical answer from GSM8K format (#### marker)"""
    if "####" not in text:
        return None
    # GSM8K uses format: "Explanation... #### 42"
    return text.split("####")[1].strip()

def process_dataset_example(example):
    """Convert GSM8K example to conversation format for GRPO training"""
    question = example["question"]
    answer = extract_hash_answer(example["answer"])

    # Create conversation with system prompt for structured reasoning
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    return {
        "prompt": prompt,           # Input conversation
        "answer": answer,          # Ground truth for reward functions
    }

print("🔄 Loading GSM8K mathematical reasoning dataset...")
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Apply conversation formatting to all examples
dataset = dataset.map(process_dataset_example)

print(f"Dataset loaded and processed!")
print(f"Training examples: {len(dataset):,}")
print(f"Sample question: {dataset[0]['prompt'][1]['content']}...")
print(f"Sample answer: {dataset[0]['answer']}")

# Show structure of first example for verification
print(f"\nExample structure:")
print(f"   Prompt: {len(dataset[0]['prompt'])} messages (system + user)")
print(f"   Answer: {dataset[0]['answer']} (ground truth for rewards)")

# Compiled regex patterns for efficient reward computation
match_format = re.compile(
    rf"^[\s]{{0,}}"                      # Optional whitespace at start
    rf"{reasoning_start}.+?{reasoning_end}.*?"  # Reasoning section (non-greedy)
    rf"{solution_start}(.+?){solution_end}"     # Solution section with capture group
    rf"[\s]{{0,}}$",                     # Optional whitespace at end
    flags=re.MULTILINE | re.DOTALL       # Multi-line matching with . matching newlines
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", # Extract numbers from solution section
    flags=re.MULTILINE | re.DOTALL        # Flexible pattern matching
)

# Reward Function 1: Exact Format Compliance
def match_format_exactly(completions, **kwargs):
    """
    High reward (3.0) for perfect format adherence
    Ensures model learns the complete structured output pattern
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        # Check if response matches complete format pattern
        score = 3.0 if match_format.search(response) is not None else 0.0
        scores.append(score)
    return scores

# Reward Function 2: Partial Format Credit
def match_format_approximately(completions, **kwargs):
    """
    Graduated scoring for format elements
    Encourages learning individual components even if not perfect
    """
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0

        # Award +0.5 for correct token count, -0.5 for wrong count
        score += 0.5 if response.count(reasoning_start) == 1 else -0.5
        score += 0.5 if response.count(reasoning_end) == 1 else -0.5
        score += 0.5 if response.count(solution_start) == 1 else -0.5
        score += 0.5 if response.count(solution_end) == 1 else -0.5

        scores.append(score)
    return scores


# Reward Function 3: Mathematical Accuracy
def check_answer_correctness(prompts, completions, answer, **kwargs):
    """
    Graduated scoring for mathematical accuracy:
    - 3.0: Exact match
    - 1.5: Within 10% (close answer)
    - 0.5: Within 20% (reasonable attempt)
    - -0.5: Wrong answer (penalty for incorrect math)
    """
    responses = [completion[0]["content"] for completion in completions]

    # Extract answers using format pattern
    extracted_responses = [
        guess.group(1) if (guess := match_format.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:  # No extractable answer
            scores.append(0)
            continue

        # Exact string match gets full points
        if guess.strip() == true_answer.strip():
            scores.append(3.0)
        else:
            # Try numerical comparison for partial credit
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:      # Within 10%
                    scores.append(1.5)
                elif 0.8 <= ratio <= 1.2:    # Within 20%
                    scores.append(0.5)
                else:                         # Wrong answer
                    scores.append(-0.5)
            except (ValueError, ZeroDivisionError):
                scores.append(-0.5)           # Invalid numerical format

    return scores

 # Reward Function 4: Number Extraction Ability
def check_numbers_extraction(prompts, completions, answer, **kwargs):
    """
    Tests the model's ability to extract numerical values from solution sections
    Complementary to exact format matching - focuses on parsing capability
    """
    responses = [completion[0]["content"] for completion in completions]

    # Extract numbers from solution sections using number pattern
    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:  # No extractable number
            scores.append(0)
            continue

        try:
            # Simple numerical equality check
            true_val = float(true_answer.strip())
            guess_val = float(guess.strip())
            # Binary scoring: correct (1.5) or incorrect (0)
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, TypeError):
            scores.append(0)  # Invalid number format

    return scores

PROJECT_NAME = "gsm8k-grpo"
HF_USER = "SArmagan"

RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

SAVE_STEPS = 5

training_args = GRPOConfig(
    learning_rate=5e-6,

    # Memory-efficient batch configuration
    per_device_train_batch_size=8,   # Small batch for GPU memory constraints
    gradient_accumulation_steps=2,   # Effective batch size = 2 * 8 = 16

    # Sequence length limits for mathematical problems
    # max_prompt_length=1024,          # Sufficient for complex word problems
    max_completion_length=1024,      # Room for detailed step-by-step reasoning

    # Training duration and monitoring
    logging_steps=1,                 # Log metrics every step for close monitoring

    # Stability and output configuration
    max_grad_norm=0.1,               # Aggressive gradient clipping for stable training

    output_dir=PROJECT_RUN_NAME,
    run_name=RUN_NAME,
    report_to="wandb",

    save_steps=SAVE_STEPS,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    eval_strategy="no",
    eval_steps=SAVE_STEPS,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

)

trainer = GRPOTrainer(
    model=model,                      # LoRA-adapted quantized model
    reward_funcs=[                    # Four complementary reward functions
        match_format_exactly,         # Perfect structure compliance
        match_format_approximately,   # Partial format credit
        check_answer_correctness,     # Mathematical accuracy
        check_numbers_extraction,     # Number parsing ability
    ],
    args=training_args,               # Training configuration
    train_dataset=dataset,            # Processed GSM8K dataset
)

trainer.train()
trainer.model.push_to_hub(PROJECT_RUN_NAME, private=True)
print(f"Saved to the hub: {PROJECT_RUN_NAME}")

wandb.finish()

# ============================================================
# GRPO MODEL EVALUATION ON GSM8K TEST SET
# ============================================================

import re
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ───────────────────────────────────────────────────
EVAL_MODEL_ID   = "SArmagan/gsm8k-grpo-2026-03-04_03.13.49" # your trained hub model, or swap for a local path
BASE_MODEL_ID   = "meta-llama/Llama-3.2-3B-Instruct"  # same base as training
BATCH_SIZE      = 8                  # reduce if OOM
MAX_NEW_TOKENS  = 1024
NUM_EXAMPLES    = None               # set e.g. 200 for a quick run, None = full test set
TEMPERATURE     = 0.0                # greedy decode for deterministic eval
# ─────────────────────────────────────────────────────────────

# ── Load test dataset ────────────────────────────────────────
print("Loading GSM8K test split...")
test_dataset = load_dataset("openai/gsm8k", "main", split="test")
if NUM_EXAMPLES:
    test_dataset = test_dataset.select(range(NUM_EXAMPLES))
print(f"  Evaluating on {len(test_dataset)} examples")

# ── Load model + adapter ─────────────────────────────────────
print(f"\nLoading base model: {BASE_MODEL_ID}")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
eval_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
print(f"Loading LoRA adapter: {EVAL_MODEL_ID}")
eval_model = PeftModel.from_pretrained(eval_model, EVAL_MODEL_ID)
eval_model.eval()

eval_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if eval_tokenizer.pad_token is None:
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
eval_tokenizer.padding_side = "left"   # required for decoder-only batch generation
print("Model ready.\n")

# ── Helpers ──────────────────────────────────────────────────
def build_prompt(question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    return eval_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def extract_gt_answer(text: str):
    """Pull ground-truth answer after GSM8K '####' marker."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")

def extract_predicted_answer(response: str):
    """Try exact format match first, fall back to last number in <SOLUTION>."""
    m = match_format.search(response)
    if m:
        return m.group(1).strip().replace(",", "")
    m2 = match_numbers.search(response)
    if m2:
        return m2.group(1).strip().replace(",", "")
    return None

def answers_match(pred: str, gt: str) -> bool:
    if pred is None or gt is None:
        return False
    if pred == gt:
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return False

# ── Evaluation loop ──────────────────────────────────────────
results = []

questions = [ex["question"] for ex in test_dataset]
gt_answers = [extract_gt_answer(ex["answer"]) for ex in test_dataset]
prompts    = [build_prompt(q) for q in questions]

print(f"Running inference (batch_size={BATCH_SIZE}) ...")
for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    batch_gts     = gt_answers[i : i + BATCH_SIZE]

    inputs = eval_tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(eval_model.device)

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=eval_tokenizer.pad_token_id,
            eos_token_id=eval_tokenizer.eos_token_id,
        )
        if TEMPERATURE > 0:
            gen_kwargs.update(do_sample=True, temperature=TEMPERATURE)
        else:
            gen_kwargs["do_sample"] = False

        output_ids = eval_model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    new_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    responses = eval_tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    for response, gt in zip(responses, batch_gts):
        pred       = extract_predicted_answer(response)
        correct    = answers_match(pred, gt)
        has_format = match_format.search(response) is not None
        results.append({
            "gt":         gt,
            "pred":       pred,
            "correct":    correct,
            "has_format": has_format,
            "response":   response,
        })

# ── Metrics ──────────────────────────────────────────────────
total        = len(results)
correct      = sum(r["correct"]    for r in results)
format_ok    = sum(r["has_format"] for r in results)
no_answer    = sum(r["pred"] is None for r in results)

accuracy     = correct   / total * 100
format_rate  = format_ok / total * 100
no_ans_rate  = no_answer / total * 100

print("\n" + "="*50)
print("         GSM8K TEST SET EVALUATION RESULTS")
print("="*50)
print(f"  Total examples   : {total}")
print(f"  Correct answers  : {correct}  ({accuracy:.2f}%)")
print(f"  Format compliant : {format_ok} ({format_rate:.2f}%)")
print(f"  No answer parsed : {no_answer} ({no_ans_rate:.2f}%)")
print("="*50)

# ── Qualitative spot-check (5 random examples) ───────────────
import random
print("\n--- Spot-check (5 random examples) ---")
for r in random.sample(results, min(5, total)):
    status = "✅" if r["correct"] else "❌"
    fmt    = "📐" if r["has_format"] else "⚠️ "
    print(f"\n{status} {fmt}  GT: {r['gt']}  |  Pred: {r['pred']}")
    print(f"   Response snippet: {r['response'][:200].strip()} ...")

# ============================================================
# BASE MODEL EVALUATION ON GSM8K TEST SET (standalone)
# ============================================================

import re
import random
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ───────────────────────────────────────────────────
BASE_MODEL_ID  = "meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE     = 16
MAX_NEW_TOKENS = 1024
NUM_EXAMPLES   = None   # set e.g. 200 for quick run, None = full 1319 test set

# ── Format tokens (must match training) ──────────────────────
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

system_prompt = f"""You are a mathematical reasoning assistant.
When given a math problem:
1. Show your step-by-step work between {reasoning_start} and {reasoning_end}
2. Provide your final numerical answer between {solution_start} and {solution_end}
3. Be precise and show all calculation steps clearly.

Example output format:
{reasoning_start}
First, I need to add 2 and 2. 2 + 2 = 4.
{reasoning_end}
{solution_start}
4
{solution_end}"""

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL
)

# ── Helpers ──────────────────────────────────────────────────
def extract_gt_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")

def extract_predicted_answer(response):
    m = match_format.search(response)
    if m:
        return m.group(1).strip().replace(",", "")
    m2 = match_numbers.search(response)
    if m2:
        return m2.group(1).strip().replace(",", "")
    return None

def answers_match(pred, gt):
    if pred is None or gt is None:
        return False
    if pred == gt:
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return False

# ── Load dataset ─────────────────────────────────────────────
print("Loading GSM8K test split...")
test_dataset = load_dataset("openai/gsm8k", "main", split="test")
if NUM_EXAMPLES:
    test_dataset = test_dataset.select(range(NUM_EXAMPLES))
print(f"  {len(test_dataset)} examples")

gt_answers = [extract_gt_answer(ex["answer"]) for ex in test_dataset]

# ── Load tokenizer + model ───────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"\nLoading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_cfg,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
base_model.eval()

# ── Build prompts ─────────────────────────────────────────────
def build_prompt(question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

prompts = [build_prompt(ex["question"]) for ex in test_dataset]

# ── Inference loop ────────────────────────────────────────────
results = []
print(f"\nRunning inference (batch_size={BATCH_SIZE}) ...")
for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
    batch_prompts = prompts[i : i + BATCH_SIZE]
    batch_gts     = gt_answers[i : i + BATCH_SIZE]

    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(base_model.device)

    with torch.no_grad():
        output_ids = base_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    new_ids   = output_ids[:, inputs["input_ids"].shape[1]:]
    responses = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    for response, gt in zip(responses, batch_gts):
        pred       = extract_predicted_answer(response)
        results.append({
            "gt":         gt,
            "pred":       pred,
            "correct":    answers_match(pred, gt),
            "has_format": match_format.search(response) is not None,
            "response":   response,
        })

# ── Metrics ───────────────────────────────────────────────────
total     = len(results)
correct   = sum(r["correct"]    for r in results)
format_ok = sum(r["has_format"] for r in results)
no_answer = sum(r["pred"] is None for r in results)

print(f"\n{'='*50}")
print(f"  BASE MODEL RESULTS")
print(f"{'='*50}")
print(f"  Total examples   : {total}")
print(f"  Correct answers  : {correct}  ({correct/total*100:.2f}%)")
print(f"  Format compliant : {format_ok} ({format_ok/total*100:.2f}%)")
print(f"  No answer parsed : {no_answer} ({no_answer/total*100:.2f}%)")
print(f"{'='*50}")

# ── Spot-check ────────────────────────────────────────────────
print("\n--- Spot-check (5 random examples) ---")
for r in random.sample(results, min(5, total)):
    status = "✅" if r["correct"] else "❌"
    fmt    = "📐" if r["has_format"] else "⚠️ "
    print(f"\n{status} {fmt}  GT: {r['gt']}  |  Pred: {r['pred']}")
    print(f"   Response snippet: {r['response'][:200].strip()} ...")