import os
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Requires access
DATASET_NAME = "databricks/databricks-dolly-15k"
OUTPUT_DIR_BASE = "/mnt/pvc/finetuned_llama3_1_8b" # Example if using PVC, or GCS path like "gs://your-bucket/output"
CHECKPOINT_DIR_BASE = "/mnt/pvc/finetune_checkpoints" # Or GCS

# QLoRA config
USE_QLORA = True
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_R = 64
BNB_4BIT_COMPUTE_DTYPE = "bfloat16" # or "float16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# Training arguments
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Adjust based on GPU memory
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = num_gpus * batch_size * grad_accum
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
GROUP_BY_LENGTH = True # Speeds up training by batching similar length sequences
MAX_SEQ_LENGTH = 1024 # Adjust based on your data and GPU memory
PACKING = False # If using SFTTrainer's packing feature

# --- Helper function to read HF token ---
def get_hf_token():
    token_path = "/etc/hf-token/HF_TOKEN" # Path defined in RayCluster volumeMount
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()
    return os.getenv("HF_TOKEN") # Fallback to environment variable

def format_dataset(example):
    # Basic formatting for Dolly dataset
    # The SFTTrainer can often infer this, but explicit formatting is safer.
    if example.get("context"):
        return f"Instruction: {example['instruction']}\nContext: {example['context']}\nResponse: {example['response']}"
    else:
        return f"Instruction: {example['instruction']}\nResponse: {example['response']}"

# --- Ray Train Training Function ---
def train_loop_per_worker(config):
    # Distributed setup
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    print(f"Starting training on rank {rank} of {world_size} workers.")

    hf_token = get_hf_token()
    if not hf_token:
        print("Hugging Face token not found. Make sure it's available as a secret or env var.")
        # Potentially raise an error or proceed if model is public & doesn't need auth for weights

    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token # Or a specific pad token if your model has one
    tokenizer.padding_side = "right" # Fix for some models

    if USE_QLORA:
        compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=USE_NESTED_QUANT,
        )
        # For training, device_map should be handled by Accelerate/FSDP if used, or set explicitly.
        # For SFTTrainer with a single GPU per worker, 'auto' or current device is fine.
        device_map = {"": train.torch.get_device()} # Assigns model to current Ray worker's GPU
    else:
        bnb_config = None
        device_map = {"": train.torch.get_device()}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if USE_QLORA else None,
        device_map=device_map, # Let Ray/Accelerate handle device placement
        trust_remote_code=True,
        token=hf_token,
        torch_dtype=compute_dtype if USE_QLORA else torch.bfloat16 # or torch.float16
    )
    model.config.use_cache = False # Necessary for LoRA
    model.config.pretraining_tp = 1 # For some models like Llama

    if USE_QLORA:
        peft_config = LoraConfig(
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            r=LORA_R,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Common for Llama
        )
    else:
        peft_config = None

    # 2. Load and prepare dataset
    # Ray Data can be used here for large datasets and distributed preprocessing
    # For simplicity, loading directly on each worker (small dataset)
    raw_dataset = load_dataset(DATASET_NAME, split="train") # Consider streaming for large datasets
    # Optional: Shuffle and select a subset for faster example run
    raw_dataset = raw_dataset.shuffle(seed=42).select(range(min(1000, len(raw_dataset))))


    # 3. Set up TrainingArguments for SFTTrainer
    # Unique output and checkpoint dirs for each run if needed
    output_dir = os.path.join(OUTPUT_DIR_BASE, f"run_{train.get_context().get_trial_name()}")
    checkpoint_dir = os.path.join(CHECKPOINT_DIR_BASE, f"run_{train.get_context().get_trial_name()}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="epoch", # Or "steps"
        # save_steps=100, # If save_strategy="steps"
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=-1, # Overridden by num_train_epochs
        # FSDP arguments if not using Accelerate's default or deepspeed
        # fsdp="full_shard auto_wrap", # Example, requires accelerate config
        # fsdp_config={"fsdp_transformer_layer_cls_to_wrap": ['LlamaDecoderLayer']},
        # bf16=True, # if using bfloat16
        # tf32=True, # if on Ampere+
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none", # Integrate W&B
        # Pushed to Hub args (optional)
        # push_to_hub=True,
        # hub_model_id=f"your_hf_username/{MODEL_ID.split('/')[-1]}-finetuned",
        # hub_token=hf_token,
    )

    # 4. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=raw_dataset,
        # dataset_text_field="text", # If your dataset is a single text column after formatting
        formatting_func=format_dataset, # Use this if your dataset needs formatting
        peft_config=peft_config if USE_QLORA else None,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=PACKING,
    )

    # 5. Start training
    print(f"Rank {rank}: Starting SFTTrainer.train()")
    train_result = trainer.train()
    print(f"Rank {rank}: Training finished.")

    # 6. Save model and tokenizer
    # Let rank 0 do the final save to avoid race conditions if not using FSDP with consolidated save
    # SFTTrainer with PEFT typically saves only the adapter.
    # If you want to save the full model, you might need to merge weights first.
    if rank == 0:
        print(f"Rank {rank}: Saving model and tokenizer to {output_dir}")
        trainer.save_model(output_dir) # Saves adapter
        tokenizer.save_pretrained(output_dir)
        print(f"Rank {rank}: Model and tokenizer saved.")

        # If you want to merge and save the full model (requires enough CPU RAM)
        # from peft import PeftModel
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     MODEL_ID,
        #     torch_dtype=torch.bfloat16, # Or float16
        #     device_map="cpu", # Load on CPU for merging
        #     trust_remote_code=True,
        #     token=hf_token
        # )
        # merged_model = PeftModel.from_pretrained(base_model, output_dir)
        # merged_model = merged_model.merge_and_unload()
        # merged_model.save_pretrained(os.path.join(output_dir, "merged_model"), safe_serialization=True)
        # tokenizer.save_pretrained(os.path.join(output_dir, "merged_model"))
        # print(f"Rank {rank}: Merged model saved to {os.path.join(output_dir, 'merged_model')}")

    # Report metrics to Ray Train
    # metrics = train_result.metrics
    # train.report(metrics)


if __name__ == "__main__":
    # --- Ray Cluster Connection & Job Submission ---
    # If running locally and connecting to a remote Ray cluster:
    # ray.init(address="ray://<ray_head_ip_or_service>:10001", runtime_env={"working_dir": "."})
    # If this script is submitted via `ray job submit` or run inside a Ray pod,
    # ray.init() or ray.init(address="auto") is usually sufficient.
    # For RayJob CRD, ray.init() is called automatically in the environment.

    num_workers = 2 # Should match replicas in RayCluster workerGroupSpecs
    use_gpu_per_worker = True

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu_per_worker,
        resources_per_worker={"GPU": 1} # Ensure this matches RayCluster worker resources
    )

    # Optional: Configure RunConfig for storage, failure handling, etc.
    # Make sure OUTPUT_DIR_BASE and CHECKPOINT_DIR_BASE are accessible by all workers (e.g., GCS or shared PVC)
    # For GCS, ensure service account has permissions.
    # Example for GCS (uncomment and configure if using GCS):
    # from ray.train.torch import TorchCheckpoint
    # run_config = RunConfig(
    #     storage_path="gs://your-ray-results-bucket/train_results", # GCS path for checkpoints and results
    #     name="llama3_finetune_job",
    #     checkpoint_config=train.CheckpointConfig(
    #         num_to_keep=2,
    #         checkpoint_score_attribute="loss", # Or other metric from trainer.evaluate()
    #         checkpoint_score_order="min"
    #     )
    # )

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={}, # Pass any static configs to the train_loop
        scaling_config=scaling_config,
        # run_config=run_config, # Uncomment if using GCS or custom run config
    )

    result = trainer.fit()

    print("Fine-tuning job completed.")
    print(f"Results: {result.metrics}")
    if result.checkpoint:
        print(f"Last checkpoint: {result.checkpoint}")