import os
import json

import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# --- Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "gretelai/synthetic_text_to_sql"
OUTPUT_DIR_BASE = "/mnt/pvc/finetuned_llama3_1_8b_gretel_sql"
# CHECKPOINT_DIR_BASE = "/mnt/pvc/finetune_checkpoints"

# QLoRA config
USE_QLORA = True
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_R = 64
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_NESTED_QUANT = False

# Training parameters (will be passed to SFTConfig)
NUM_TRAIN_EPOCHS = 1
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
REPORT_TO = "tensorboard"

# SFT specific parameters (will be passed to SFTConfig)
MAX_SEQ_LENGTH = 1024
PACKING = False
GROUP_BY_LENGTH = True

def run_inference_comparison(
    original_model_id: str,
    path_to_fine_tuned_model: str,
    path_to_tokenizer: str,
    eval_dataset_name: str,
    num_eval_samples: int,
    max_seq_len: int, # Overall sequence length limit
    results_output_dir: str,
    hf_auth_token: str,
    device: torch.device,
    max_new_generation_tokens: int = 150
):
    print(f"\n[Rank 0] Starting inference comparison on device: {device}")

    # 1. Load Tokenizer
    print(f"[Rank 0] Loading tokenizer from {path_to_tokenizer}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # For Llama 3, padding_side='left' is often recommended for batched generation.
        # For single sample generation here, it's less critical but good to be aware.
        # The training used "right", so let's stick to that for tokenizing the prompt itself for now.
        tokenizer.padding_side = "right"
    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load tokenizer from {path_to_tokenizer}: {e}")
        return

    # 2. Load Fine-tuned Model
    print(f"[Rank 0] Loading fine-tuned model from {path_to_fine_tuned_model}...")
    try:
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            path_to_fine_tuned_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        fine_tuned_model.to(device)
        fine_tuned_model.eval()
        print(f"[Rank 0] Fine-tuned model dtype: {fine_tuned_model.dtype}")
        # print(f"[Rank 0] Fine-tuned model config: {fine_tuned_model.config}") # For deeper inspection if needed
    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load fine-tuned model from {path_to_fine_tuned_model}: {e}")
        return

    # 3. Load Original Pre-trained Model
    print(f"[Rank 0] Loading original pre-trained model {original_model_id}...")
    try:
        original_model = AutoModelForCausalLM.from_pretrained(
            original_model_id,
            token=hf_auth_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        original_model.to(device)
        original_model.eval()
        print(f"[Rank 0] Original model dtype: {original_model.dtype}")
    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load original model {original_model_id}: {e}")
        del fine_tuned_model
        torch.cuda.empty_cache()
        return

    # 4. Prepare Inference Data
    print(f"[Rank 0] Loading {num_eval_samples} raw samples from '{eval_dataset_name}' (test split) for inference...")
    try:
        target_sql_complexity = "window functions"

        # Load the full test split
        full_test_dataset = load_dataset(
            eval_dataset_name,
            split="test",
            token=hf_auth_token,
            trust_remote_code=True # Keep if dataset might need it
        )
        
        # Apply the filter directly
        filtered_dataset = full_test_dataset.filter(
            lambda example: example.get('sql_complexity', '').lower() == target_sql_complexity.lower()
        )
        
        eval_samples_for_inference = filtered_dataset.select(range(num_eval_samples))

    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load samples from {eval_dataset_name} for inference: {e}")
        del original_model, fine_tuned_model
        torch.cuda.empty_cache()
        return

    # 5. Define Generation Utilities
    def generate_sql_with_chat_template(current_model, current_tokenizer, sql_context_str, sql_prompt_str):
        system_prompt_content = "You are a precise SQL query generation assistant. Given a database schema and a user question, you MUST generate only the SQL query that directly answers the question. Do not include any other explanatory text, markdown formatting, or any conversational preamble."
        user_content = f"### Database Schema:\n{sql_context_str}\n\n### User Question:\n{sql_prompt_str}\n\n### SQL Query:"
        # The "### SQL Query:" in user_content explicitly asks for the SQL.
        
        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_content}
            # Assistant turn is what the model will generate
        ]
        
        # `add_generation_prompt=True` appends the start of the assistant's turn, e.g., <|start_header_id|>assistant<|end_header_id|>
        input_prompt_for_model = current_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True 
        )
        
        input_max_len = max_seq_len - max_new_generation_tokens
        if input_max_len <= 0: input_max_len = max(50, max_seq_len // 2)

        inputs = current_tokenizer(input_prompt_for_model, return_tensors="pt", truncation=True, max_length=input_max_len).to(current_model.device)

        if 'token_type_ids' in inputs: del inputs['token_type_ids']

        generation_output = current_model.generate(
            **inputs, 
            max_new_tokens=max_new_generation_tokens, 
            pad_token_id=current_tokenizer.eos_token_id,
            # Llama 3 uses specific EOT tokens. Tokenizer should handle this via chat template.
            # Forcing a specific list might be good:
            eos_token_id=[current_tokenizer.eos_token_id, current_tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            do_sample=False,
            temperature=1.0, # Default for do_sample=False, suppresses UserWarning
            top_p=None       # Default for do_sample=False, suppresses UserWarning
        )
        response_ids = generation_output[0][inputs['input_ids'].shape[1]:]
        response_text = current_tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text.strip(), input_prompt_for_model # Return the template-formatted prompt

    # Loop, Generate, Print, and Collect (using the new unified generation function)
    all_inference_results = []
    print("\n--- [Rank 0] Inference Comparison START ---")
    for i, sample in enumerate(eval_samples_for_inference):
        # --- CORRECTED EXTRACTION ---
        # Gretel dataset fields: 'id', 'sql_context', 'sql_prompt', 'sql'
        sample_id_val = sample.get('id', f"unknown_id_{i}") # Use a default if 'id' is missing
        sql_context_str = sample.get('sql_context', '')    # Use .get for safety, provide default
        sql_prompt_str = sample.get('sql_prompt', '')    # Use .get for safety
        ground_truth_sql_str = sample.get('sql', '')        # Use .get for safety
        # --- END OF CORRECTION ---

        print(f"\n--- [Rank 0] Sample {i+1}/{num_eval_samples} (ID: {sample_id_val}) ---")
        print(f"SCHEMA (Context):\n{sql_context_str}")
        print(f"QUESTION (Prompt):\n{sql_prompt_str}")
        print(f"\nGROUND TRUTH SQL:\n{ground_truth_sql_str}")

        # Now sql_context_str and sql_prompt_str are defined
        orig_response_text, orig_input_prompt = generate_sql_with_chat_template(original_model, tokenizer, sql_context_str, sql_prompt_str)
        print(f"\nORIGINAL MODEL (Llama 3 Chat Template):\n{orig_response_text}")
        
        ft_response_text, ft_input_prompt = generate_sql_with_chat_template(fine_tuned_model, tokenizer, sql_context_str, sql_prompt_str)
        print(f"\nFINE-TUNED MODEL (Llama 3 Chat Template):\n{ft_response_text}")
        print("-------------------------------")

        all_inference_results.append({
             "id": sample_id_val, # Use the extracted id
             "schema_context": sql_context_str,
             "question_prompt": sql_prompt_str,
             "original_model_input_prompt": orig_input_prompt, 
             "fine_tuned_model_input_prompt": ft_input_prompt,
             "ground_truth_sql": ground_truth_sql_str,
             "original_model_sql_response": orig_response_text, 
             "fine_tuned_model_sql_response": ft_response_text
        })

    # 7. Save to GCS
    inference_output_filename = "inference_comparison_results_v2.json" # New filename
    inference_output_path = os.path.join(results_output_dir, inference_output_filename)
    try:
        with open(inference_output_path, 'w') as f:
            json.dump(all_inference_results, f, indent=4)
        print(f"[Rank 0] Inference comparison results saved to: {inference_output_path} (on GCS)")
    except Exception as e:
        print(f"[Rank 0] ERROR saving inference results to {inference_output_path}: {e}")
    
    # 8. Clean up models
    del original_model, fine_tuned_model
    torch.cuda.empty_cache()
    print(f"[Rank 0] Cleaned up inference models from memory on device: {device}.")


# --- Ray Train Training Function ---
def train_loop_per_worker(config):

    # Determine current rank and world size.
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    print(f"Starting training on rank {rank} of {world_size} workers.")
    
    # Extract HF token for tokenizer, model, and data retrieval.
    hf_token = os.getenv("HF_TOKEN")

    # Download and configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Rank 0 creates output directory (mounted to GCS)
    if rank == 0:
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)

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

    def format_gretel_sql_for_sft_chat_template(sample): # New name for clarity
        context_schema = sample.get('sql_context', '') or ''
        question_text = sample.get('sql_prompt', '') or ''
        sql_answer = sample.get('sql', '') or ''

        # NEW: Format using Llama 3 chat template for SFT
        # This system prompt is focused and tells the model its role.
        system_prompt_content = "You are a precise SQL query generation assistant. Given a database schema and a user question, you MUST generate only the SQL query that directly answers the question. Do not include any other explanatory text, markdown formatting, or any conversational preamble."
        
        user_content = f"### Database Schema:\n{context_schema}\n\n### User Question:\n{question_text}\n\n### SQL Query:"
        # The "### SQL Query:" part in the user message acts as a final prompt for the assistant.
        
        messages_for_sft = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sql_answer} # Ground truth SQL
        ]
        
        # SFTTrainer will tokenize the "text" field.
        # `apply_chat_template` adds BOS, EOS, and all special Llama 3 tokens.
        # `add_generation_prompt=False` because we provide the assistant's complete response for training.
        formatted_text = tokenizer.apply_chat_template(
            messages_for_sft, 
            tokenize=False, 
            add_generation_prompt=False 
        )
        return {"text": formatted_text}

    print(f"Rank {rank}: Loading dataset '{DATASET_NAME}'...")
    # Load train and test splits
    train_dataset_raw = load_dataset(DATASET_NAME, split="train", token=hf_token)
    eval_dataset_raw = load_dataset(DATASET_NAME, split="test", token=hf_token)
    print(f"Rank {rank}: Loaded raw train split with {len(train_dataset_raw)} samples and test split with {len(eval_dataset_raw)} samples.")

    # Apply formatting
    train_original_columns = list(train_dataset_raw.features)
    eval_original_columns = list(eval_dataset_raw.features)

    print(f"Rank {rank}: Formatting datasets using Llama 3 chat template for SFT...")

    train_dataset = train_dataset_raw.map(format_gretel_sql_for_sft_chat_template, remove_columns=train_original_columns)
    eval_dataset = eval_dataset_raw.map(format_gretel_sql_for_sft_chat_template, remove_columns=eval_original_columns)
    
    # Shuffle and select subsets (adjust as needed)
    # For a real run, you might use more or all of the data.
    train_dataset = train_dataset.shuffle(seed=42).select(range(min(1000, len(train_dataset))))
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(200, len(eval_dataset)))) # Using a smaller subset for faster eval

    # 3. Set up SFTConfig
    sft_output_dir = os.path.join(OUTPUT_DIR_BASE, "sft_model_output_sql_gretel")

    sft_config = SFTConfig(
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
        packing=PACKING,
        dataset_text_field="text",
        evaluation_strategy="steps",
        eval_steps=30
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config_instance
    )

    print(f"Rank {rank}: Starting SFTTrainer.train()")
    train_result = trainer.train()
    print(f"Rank {rank}: Training finished. Metrics: {train_result.metrics}")

    if train.get_context().get_world_rank() == 0:
        print(f"Rank {rank}: SFTTrainer training complete. Final model/adapter saved to {sft_config.output_dir}.")
        
        path_to_saved_fine_tuned_model = None
        # This path should point to where your *final, usable* fine-tuned model is saved
        # (either the merged QLoRA model or the fully fine-tuned model)
        
        if USE_QLORA:
            # This path is where merge_and_unload saved the merged model
            path_to_saved_fine_tuned_model = os.path.join(OUTPUT_DIR_BASE, "final_merged_model_on_gcs")
            # The tokenizer was also saved here in the previous step
            path_to_saved_tokenizer = path_to_saved_fine_tuned_model
            
            # Ensure the merge_and_unload logic from previous step has run and created this path
            # (The merge and save logic should be here, before calling inference)
            print(f"Rank {rank}: Merging adapter and saving full model to {path_to_saved_fine_tuned_model}...")
            model_to_merge = None
            if hasattr(trainer.model, 'merge_and_unload'): model_to_merge = trainer.model
            elif hasattr(trainer.model, 'module') and hasattr(trainer.model.module, 'merge_and_unload'): model_to_merge = trainer.model.module
            else: raise AttributeError("Cannot find merge_and_unload on trainer.model or trainer.model.module.")
            
            merged_model = model_to_merge.merge_and_unload()
            merged_model.save_pretrained(path_to_saved_fine_tuned_model)
            tokenizer.save_pretrained(path_to_saved_tokenizer) # Save tokenizer with merged model
            print(f"Rank {rank}: Full merged model (and tokenizer) saved to {path_to_saved_fine_tuned_model}.")
            del merged_model # Free up memory if it's reloaded in the function, though we pass the path
            torch.cuda.empty_cache()

        else: # Full fine-tuning
            path_to_saved_fine_tuned_model = os.path.join(OUTPUT_DIR_BASE, "final_model_on_gcs") # Or your specific full ft save path
            path_to_saved_tokenizer = path_to_saved_fine_tuned_model

            print(f"Rank {rank}: Saving fully fine-tuned model to {path_to_saved_fine_tuned_model}...")
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            model_to_save.save_pretrained(path_to_saved_fine_tuned_model)
            tokenizer.save_pretrained(path_to_saved_tokenizer) # Save tokenizer
            print(f"Rank {rank}: Full model (and tokenizer) saved to {path_to_saved_fine_tuned_model}.")
            del model_to_save # Free up memory
            torch.cuda.empty_cache()

        # --- Call the new inference function ---
        if os.path.exists(path_to_saved_fine_tuned_model) and os.path.exists(path_to_saved_tokenizer):
            run_inference_comparison(
                original_model_id=MODEL_ID,    # Your global MODEL_ID
                path_to_fine_tuned_model=path_to_saved_fine_tuned_model,
                path_to_tokenizer=path_to_saved_tokenizer, # Tokenizer saved with fine-tuned model
                eval_dataset_name=DATASET_NAME, # Your global DATASET_NAME
                num_eval_samples=2,             # Or make this configurable
                max_seq_len=MAX_SEQ_LENGTH,     # Your global MAX_SEQ_LENGTH
                results_output_dir=OUTPUT_DIR_BASE, # Your global OUTPUT_DIR_BASE
                hf_auth_token=hf_token,         # hf_token from train_loop_per_worker's scope
                device=train.torch.get_device(), # Get current worker's device
                max_new_generation_tokens=300   # Adjust as needed for SQL query length
            )
        else:
            print(f"[Rank 0] ERROR: Saved fine-tuned model or tokenizer path not found, skipping inference comparison.")
            print(f"[Rank 0] Checked path for model: {path_to_saved_fine_tuned_model}")
            print(f"[Rank 0] Checked path for tokenizer: {path_to_saved_tokenizer}")

if __name__ == "__main__":
    num_gpus_per_worker = 1
    num_workers_ray_train = 16

    scaling_config = ScalingConfig(
        num_workers=num_workers_ray_train,
        use_gpu=True,
        resources_per_worker={"GPU": num_gpus_per_worker}
    )

    torch_trainer_instance = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={},
        scaling_config=scaling_config,
    )

    result = torch_trainer_instance.fit()

    print("Fine-tuning job completed.")
    print(f"Results: {result.metrics}")
    if result.checkpoint:
        print(f"Last Ray Train checkpoint (ephemeral if not synced): {result.checkpoint}")

    sft_output_final_dir = os.path.join(OUTPUT_DIR_BASE, "sft_model_output_on_gcs")
    print(f"SFTTrainer outputs (model adapters, checkpoints) were saved to: {sft_output_final_dir}")