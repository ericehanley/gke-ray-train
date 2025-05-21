import os
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig # RunConfig not used in this version
from ray.train.torch import TorchTrainer
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments, # We will use its parameters for SFTConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig # Ensure SFTConfig is imported

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "philschmid/dolly-15k-oai-style"
OUTPUT_DIR_BASE = "/mnt/pvc/finetuned_llama3_1_8b"
CHECKPOINT_DIR_BASE = "/mnt/pvc/finetune_checkpoints"

# QLoRA config
USE_QLORA = True
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_R = 64
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# Training parameters (will be passed to SFTConfig)
NUM_TRAIN_EPOCHS = 5
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch" # "steps" or "epoch"
# SAVE_STEPS = 50 # if save_strategy is "steps"
REPORT_TO = "wandb" if os.getenv("WANDB_API_KEY") else "none"

# SFT specific parameters (will be passed to SFTConfig)
MAX_SEQ_LENGTH = 1024
PACKING = False
GROUP_BY_LENGTH = True # This is a TrainingArguments parameter

# --- Helper function to read HF token ---
def get_hf_token():
    token_path = "/etc/hf-token/HF_TOKEN"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()
    hf_token_env = os.getenv("HF_TOKEN")
    if hf_token_env:
        return hf_token_env
    print("Warning: Hugging Face token not found.")
    return None

# --- Ray Train Training Function ---
def train_loop_per_worker(config):
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    print(f"Starting training on rank {rank} of {world_size} workers.")

    hf_token = get_hf_token()
    if not hf_token and "meta-llama" in MODEL_ID:
        raise ValueError(f"Hugging Face token required for model {MODEL_ID}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if USE_QLORA:
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=USE_NESTED_QUANT,
        )
        device_map = {"": train.torch.get_device()}
    else:
        bnb_config = None
        device_map = {"": train.torch.get_device()}

    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": True,
        "token": hf_token,
        "torch_dtype": getattr(torch, BNB_4BIT_COMPUTE_DTYPE) if USE_QLORA else torch.bfloat16
    }
    if not USE_QLORA: # Avoid passing None if not used, some models warn
        model_kwargs.pop("quantization_config", None)


    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.config.use_cache = False

    if USE_QLORA:
        # For Llama 3, target_modules are important.
        llama_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        peft_config_instance = LoraConfig(
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            r=LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=llama_target_modules
        )
    else:
        peft_config_instance = None

    raw_dataset = load_dataset(DATASET_NAME, split="train")
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(min(1000, len(raw_dataset))))

    # 3. Set up SFTConfig
    sft_output_dir = os.path.join(OUTPUT_DIR_BASE, "sft_model_output_ephemeral")

    sft_config = SFTConfig(
        # Core training parameters (previously in TrainingArguments)
        output_dir=sft_output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=WEIGHT_DECAY,
        report_to=REPORT_TO,
        bf16=(BNB_4BIT_COMPUTE_DTYPE == "bfloat16" and USE_QLORA), # Enable if compute dtype is bf16
        fp16=(BNB_4BIT_COMPUTE_DTYPE == "float16" and USE_QLORA), # Enable if compute dtype is fp16
        group_by_length=GROUP_BY_LENGTH,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=PACKING
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw_dataset,
        eval_dataset=None,
        peft_config=peft_config_instance
    )

    print(f"Rank {rank}: Starting SFTTrainer.train()")
    train_result = trainer.train()
    print(f"Rank {rank}: Training finished. Metrics: {train_result.metrics}")

    if train.get_context().get_world_rank() == 0:
        print(f"Rank {rank}: Model adapter (ephemeral) potentially saved by SFTTrainer to {sft_config.output_dir}")


if __name__ == "__main__":
    num_gpus_per_worker = 1
    num_workers_ray_train = 8

    scaling_config = ScalingConfig(
        num_workers=num_workers_ray_train,
        use_gpu=True,
        resources_per_worker={"GPU": num_gpus_per_worker}
    )

    torch_trainer_instance = TorchTrainer( # Renamed to avoid conflict with SFTTrainer instance
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={},
        scaling_config=scaling_config,
    )

    result = torch_trainer_instance.fit()

    print("Fine-tuning job completed.")
    print(f"Results: {result.metrics}")
    if result.checkpoint:
        print(f"Last Ray Train checkpoint (ephemeral if not synced): {result.checkpoint}")

    sft_output_final_dir = os.path.join(OUTPUT_DIR_BASE, "sft_model_output_ephemeral")
    print(f"SFTTrainer outputs were in an ephemeral directory: {sft_output_final_dir}")
    print("WARNING: All model artifacts saved to /mnt/pvc/... were stored on ephemeral disk and are now lost unless explicitly uploaded elsewhere by the script.")