import os
import json

import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.train.huggingface.transformers import RayTrainReportCallback, prepare_trainer


import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

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
    """
    Note - this is just a basic inference comparison between tuned model and original Llama.
    It can be skipped by setting INFERENCE: false in config.json.
    """
    print(f"\n[Rank 0] Starting inference comparison on device: {device}")

    # 1. Load Tokenizer
    print(f"[Rank 0] Loading tokenizer from {path_to_tokenizer}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load tokenizer from {path_to_tokenizer}: {e}")
        return

    # 2. Load Fine-tuned Model
    print(f"[Rank 0] Loading fine-tuned model from {path_to_fine_tuned_model}...")
    try:
        fine_tuned_model = AutoModelForCausalLM.from_pretrained(
            path_to_fine_tuned_model,
            torch_dtype=torch.bfloat16, # Consider making this configurable if needed
            trust_remote_code=True
        )
        fine_tuned_model.to(device)
        fine_tuned_model.eval()
        print(f"[Rank 0] Fine-tuned model dtype: {fine_tuned_model.dtype}")
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
            torch_dtype=torch.bfloat16, # Consider making this configurable
        )
        original_model.to(device)
        original_model.eval()
        print(f"[Rank 0] Original model dtype: {original_model.dtype}")
    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load original model {original_model_id}: {e}")
        if 'fine_tuned_model' in locals(): del fine_tuned_model
        torch.cuda.empty_cache()
        return

    # 4. Prepare Inference Data
    print(f"[Rank 0] Loading {num_eval_samples} raw samples from '{eval_dataset_name}' (test split) for inference...")
    try:
        target_sql_complexity = "window functions"
        full_test_dataset = load_dataset(
            eval_dataset_name,
            split="test",
            token=hf_auth_token,
            trust_remote_code=True
        )
        filtered_dataset = full_test_dataset.filter(
            lambda example: example.get('sql_complexity', '').lower() == target_sql_complexity.lower()
        )
        if len(filtered_dataset) < num_eval_samples:
            print(f"[Rank 0] WARNING: Requested {num_eval_samples} but only found {len(filtered_dataset)} with complexity '{target_sql_complexity}'. Using all available.")
            eval_samples_for_inference = filtered_dataset
            num_eval_samples = len(filtered_dataset)
        else:
            eval_samples_for_inference = filtered_dataset.select(range(num_eval_samples))

    except Exception as e:
        print(f"[Rank 0] ERROR: Could not load samples from {eval_dataset_name} for inference: {e}")
        if 'original_model' in locals(): del original_model
        if 'fine_tuned_model' in locals(): del fine_tuned_model
        torch.cuda.empty_cache()
        return
    
    if num_eval_samples == 0:
        print(f"[Rank 0] No samples found for inference with target complexity. Skipping inference.")
        if 'original_model' in locals(): del original_model
        if 'fine_tuned_model' in locals(): del fine_tuned_model
        torch.cuda.empty_cache()
        return


    # 5. Define Generation Utilities
    def generate_sql_with_chat_template(current_model, current_tokenizer, sql_context_str, sql_prompt_str):
        system_prompt_content = "You are a precise SQL query generation assistant. Given a database schema and a user question, you MUST generate only the SQL query that directly answers the question. Do not include any other explanatory text, markdown formatting, or any conversational preamble."
        user_content = f"### Database Schema:\n{sql_context_str}\n\n### User Question:\n{sql_prompt_str}\n\n### SQL Query:"
        messages = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_content}
        ]
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
            eos_token_id=[current_tokenizer.eos_token_id, current_tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            do_sample=False,
            temperature=1.0,
            top_p=None
        )
        response_ids = generation_output[0][inputs['input_ids'].shape[1]:]
        response_text = current_tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text.strip(), input_prompt_for_model

    all_inference_results = []
    print("\n--- [Rank 0] Inference Comparison START ---")
    for i, sample in enumerate(eval_samples_for_inference):
        sample_id_val = sample.get('id', f"unknown_id_{i}")
        sql_context_str = sample.get('sql_context', '')
        sql_prompt_str = sample.get('sql_prompt', '')
        ground_truth_sql_str = sample.get('sql', '')

        print(f"\n--- [Rank 0] Sample {i+1}/{num_eval_samples} (ID: {sample_id_val}) ---")
        print(f"SCHEMA (Context):\n{sql_context_str}")
        print(f"QUESTION (Prompt):\n{sql_prompt_str}")
        print(f"\nGROUND TRUTH SQL:\n{ground_truth_sql_str}")

        orig_response_text, orig_input_prompt = generate_sql_with_chat_template(original_model, tokenizer, sql_context_str, sql_prompt_str)
        print(f"\nORIGINAL MODEL (Llama 3 Chat Template):\n{orig_response_text}")

        ft_response_text, ft_input_prompt = generate_sql_with_chat_template(fine_tuned_model, tokenizer, sql_context_str, sql_prompt_str)
        print(f"\nFINE-TUNED MODEL (Llama 3 Chat Template):\n{ft_response_text}")
        print("-------------------------------")

        all_inference_results.append({
             "id": sample_id_val,
             "schema_context": sql_context_str,
             "question_prompt": sql_prompt_str,
             "original_model_input_prompt": orig_input_prompt,
             "fine_tuned_model_input_prompt": ft_input_prompt,
             "ground_truth_sql": ground_truth_sql_str,
             "original_model_sql_response": orig_response_text,
             "fine_tuned_model_sql_response": ft_response_text
        })

    inference_output_filename = "inference_comparison_results.json" # Simpler name
    inference_output_path = os.path.join(results_output_dir, inference_output_filename)
    try:
        with open(inference_output_path, 'w') as f:
            json.dump(all_inference_results, f, indent=4)
        print(f"[Rank 0] Inference comparison results saved to: {inference_output_path} (on GCS)")
    except Exception as e:
        print(f"[Rank 0] ERROR saving inference results to {inference_output_path}: {e}")

    if 'original_model' in locals(): del original_model
    if 'fine_tuned_model' in locals(): del fine_tuned_model
    torch.cuda.empty_cache()
    print(f"[Rank 0] Cleaned up inference models from memory on device: {device}.")


# --- Ray Train Training Function ---
def train_loop_per_worker(config: dict):

    # Discover the world
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    print(f"Starting training on rank {rank} of {world_size} workers.")

    # Load tokenizer from HF
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL_ID"], trust_remote_code=True, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Create output directory
    if rank == 0:
        os.makedirs(config["OUTPUT_DIR_BASE"], exist_ok=True)

    # Configure LORA
    if config["USE_QLORA"]:
        compute_dtype = getattr(torch, config["BNB_4BIT_COMPUTE_DTYPE"])
        bnb_config_params = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config["BNB_4BIT_QUANT_TYPE"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config["USE_NESTED_QUANT"],
        )
        device_map = {"": train.torch.get_device()}
    else:
        bnb_config_params = None
        device_map = {"": train.torch.get_device()}

    # Load model from HF and provide model_kwargs
    model_kwargs = {
        "quantization_config": bnb_config_params,
        "device_map": device_map,
        "trust_remote_code": True,
        "token": hf_token,
        "torch_dtype": getattr(torch, config["BNB_4BIT_COMPUTE_DTYPE"]) if config["USE_QLORA"] else torch.bfloat16
    }
    if not config["USE_QLORA"]:
        model_kwargs.pop("quantization_config", None)

    model = AutoModelForCausalLM.from_pretrained(config["MODEL_ID"], **model_kwargs)
    model.config.use_cache = False

    # Finalize QLORA configuration
    if config["USE_QLORA"]:
        peft_config_instance = LoraConfig(
            lora_alpha=config["LORA_ALPHA"],
            lora_dropout=config["LORA_DROPOUT"],
            r=config["LORA_R"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=config["LLAMA_TARGET_MODULES"]
        )
    else:
        peft_config_instance = None

    # Helper function to format dataset to LLama friendly form
    def format_gretel_sql_for_sft_chat_template(sample):
        context_schema = sample.get('sql_context', '')
        question_text = sample.get('sql_prompt', '')
        sql_answer = sample.get('sql', '')
        system_prompt_content = "You are a precise SQL query generation assistant. Given a database schema and a user question, you MUST generate only the SQL query that directly answers the question. Do not include any other explanatory text, markdown formatting, or any conversational preamble."
        user_content = f"### Database Schema:\n{context_schema}\n\n### User Question:\n{question_text}\n\n### SQL Query:"
        messages_for_sft = [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sql_answer}
        ]
        formatted_text = tokenizer.apply_chat_template(
            messages_for_sft,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}

    # Load dataset
    print(f"Rank {rank}: Loading dataset '{config['DATASET_NAME']}'...")
    train_dataset_raw = load_dataset(config["DATASET_NAME"], split="train", token=hf_token)
    eval_dataset_raw = load_dataset(config["DATASET_NAME"], split="test", token=hf_token)
    print(f"Rank {rank}: Loaded raw train split with {len(train_dataset_raw)} samples and test split with {len(eval_dataset_raw)} samples.")

    # Extract columns
    train_original_columns = list(train_dataset_raw.features)
    eval_original_columns = list(eval_dataset_raw.features)
    print(f"Rank {rank}: Formatting datasets using Llama 3 chat template for SFT...")

    # Downsample to run faster
    # Consider making these sample counts configurable
    train_dataset = train_dataset_raw.shuffle(seed=42).select(range(min(1000, len(train_dataset_raw))))
    eval_dataset = eval_dataset_raw.shuffle(seed=42).select(range(min(200, len(eval_dataset_raw))))

    # Data processing
    train_dataset = train_dataset.map(format_gretel_sql_for_sft_chat_template, remove_columns=train_original_columns)
    eval_dataset = eval_dataset.map(format_gretel_sql_for_sft_chat_template, remove_columns=eval_original_columns)

    # Create SFT Config from config.json values
    sft_output_dir = os.path.join(config["OUTPUT_DIR_BASE"], config["SFT_SUBDIR_NAME"])
    sft_args = SFTConfig(
        num_train_epochs=config["NUM_TRAIN_EPOCHS"],
        per_device_train_batch_size=config["PER_DEVICE_TRAIN_BATCH_SIZE"],
        gradient_accumulation_steps=config["GRADIENT_ACCUMULATION_STEPS"],
        optim=config["OPTIM"],
        learning_rate=config["LEARNING_RATE"],
        lr_scheduler_type=config["LR_SCHEDULER_TYPE"],
        warmup_ratio=config["WARMUP_RATIO"],
        max_grad_norm=config["MAX_GRAD_NORM"],
        weight_decay=config["WEIGHT_DECAY"],
        bf16=(config["BNB_4BIT_COMPUTE_DTYPE"] == "bfloat16" and config["USE_QLORA"]),
        fp16=(config["BNB_4BIT_COMPUTE_DTYPE"] == "float16" and config["USE_QLORA"]),
        group_by_length=config["GROUP_BY_LENGTH"],
        max_seq_length=config["MAX_SEQ_LENGTH"],
        packing=config["PACKING"],
        dataset_text_field="text",
        output_dir=sft_output_dir,
        logging_steps=config["LOGGING_STEPS"],
        save_strategy=config["SAVE_STRATEGY"],
        report_to=config["REPORT_TO"],
        evaluation_strategy=config["EVALUATION_STRATEGY_SFT"],
        eval_steps=config["EVAL_STEPS_SFT"],
        save_steps=config["SAVE_STEPS_SFT"],
    )

    # Instantiate trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config_instance
    )

    # Distributed training
    print(f"Rank {rank}: Starting SFTTrainer.train()")
    train_result = trainer.train()
    print(f"Rank {rank}: Training finished. Metrics: {train_result.metrics}")

    # Save fine tuned model
    if rank == 0:
        print(f"Rank {rank}: SFTTrainer training complete. Final model/adapter saved to {sft_args.output_dir}.")
        saved_model_path_for_inference = None
        saved_tokenizer_path_for_inference = None

        if config["USE_QLORA"]:
            path_to_save_merged_model = os.path.join(config["OUTPUT_DIR_BASE"], config["MERGED_MODEL_SUBDIR_NAME"])
            saved_tokenizer_path_for_inference = path_to_save_merged_model # Tokenizer saved with merged model

            print(f"Rank {rank}: Merging adapter and saving full model to {path_to_save_merged_model}...")
            model_to_merge = None
            try:
                if hasattr(trainer.model, 'merge_and_unload'): model_to_merge = trainer.model
                elif hasattr(trainer.model, 'module') and hasattr(trainer.model.module, 'merge_and_unload'): model_to_merge = trainer.model.module
                else: raise AttributeError("Cannot find merge_and_unload on trainer.model or trainer.model.module.")

                merged_model = model_to_merge.merge_and_unload()
                merged_model.save_pretrained(path_to_save_merged_model)
                tokenizer.save_pretrained(saved_tokenizer_path_for_inference)
                print(f"Rank {rank}: Full merged model (and tokenizer) saved to {path_to_save_merged_model}.")
                saved_model_path_for_inference = path_to_save_merged_model
                del merged_model
                torch.cuda.empty_cache()
            except AttributeError as e:
                print(f"[Rank 0] ERROR: Could not merge and unload QLoRA model: {e}. Adapters are in {sft_args.output_dir}")
                # If merge fails, inference can't run on the merged model.
                # You might choose to load adapters directly if inference supported it,
                # or just skip inference. For now, paths remain None.


        else: # Full fine-tuning
            path_to_save_full_ft_model = os.path.join(config["OUTPUT_DIR_BASE"], config["FULL_FT_MODEL_SUBDIR_NAME"])
            saved_tokenizer_path_for_inference = path_to_save_full_ft_model

            print(f"Rank {rank}: Saving fully fine-tuned model to {path_to_save_full_ft_model}...")
            model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            model_to_save.save_pretrained(path_to_save_full_ft_model)
            tokenizer.save_pretrained(saved_tokenizer_path_for_inference)
            print(f"Rank {rank}: Full model (and tokenizer) saved to {path_to_save_full_ft_model}.")
            saved_model_path_for_inference = path_to_save_full_ft_model
            del model_to_save
            torch.cuda.empty_cache()

        # Complete side by side inference comparison
        if config["INFERENCE"]:
            if saved_model_path_for_inference and saved_tokenizer_path_for_inference and \
            os.path.exists(saved_model_path_for_inference) and os.path.exists(saved_tokenizer_path_for_inference):
                run_inference_comparison(
                    original_model_id=config["MODEL_ID"],
                    path_to_fine_tuned_model=saved_model_path_for_inference,
                    path_to_tokenizer=saved_tokenizer_path_for_inference,
                    eval_dataset_name=config["DATASET_NAME"],
                    num_eval_samples=config["NUM_EVAL_SAMPLES_INFERENCE"],
                    max_seq_len=config["MAX_SEQ_LENGTH"],
                    results_output_dir=config["OUTPUT_DIR_BASE"],
                    hf_auth_token=hf_token,
                    device=train.torch.get_device(),
                    max_new_generation_tokens=config["MAX_NEW_GENERATION_TOKENS_INFERENCE"]
                )
            else:
                print(f"[Rank 0] ERROR: Saved fine-tuned model or tokenizer path not found or not created, skipping inference comparison.")
                if saved_model_path_for_inference: print(f"[Rank 0] Checked path for model: {saved_model_path_for_inference}")
                if saved_tokenizer_path_for_inference: print(f"[Rank 0] Checked path for tokenizer: {saved_tokenizer_path_for_inference}")


if __name__ == "__main__":

    """
    Process Overview:

    1.) Extract model configuration from config.json.
    2.) Extract node/worker set up from env
    3.) Define ScalingConfig
    4.) Define TorchTrainer by passing in train_loop_per_worker, scaling_config, and train_loop_config
    5.) torch_trainer.fit()
        I.) Ray attempts to acquire num_available_workers.
        II.) Ray sets additional env variables that torch.distributed.init_process_group() uses for initialization later.
            - MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK, etc.
        III.) TorchTrainer executes train_loop_per_worker() on each allocated worker process with train_loop_config passed to each worker.
        IV.) train_loop_per_worker() progresses on each worker until SFTTrainer is instantiated.
            i.) At this point, parent class Trainer and HF Accelerate detect the distributed environment from environment variables set by Ray.
            ii.) Accelerate calls torch.distributed.init_process_group(...) using provided env variables - this is a blocking collective operation.
            iii.) NCCL initializes communication channels between GPUs and a default process group for this worker is registered in PyTorch.
        V.) trainer.train()
            i.) SFTTrainer (via Accelerate) wraps PyTorch model with torch.nn.parallel.DistributedDataParallel(model,...) which uses the default process
            group previously initialized.
            ii.) Distributed training
    6.) Fin
    """

    # Load config.json
    try:
        with open("ray-jobs/fine_tune_config.json", 'r') as f:
            loop_config = json.load(f)
    except FileNotFoundError:
        print("ERROR: config.json not found. Please create it.")
        exit(1)
    except json.JSONDecodeError:
        print("ERROR: config.json is not valid JSON.")
        exit(1)

    # Load environment/infra variables
    num_raycluster_workers = int(os.getenv("NUM_NODES", "1"))
    num_gpus_per_raycluster_worker = int(os.getenv("NUM_GPUS_PER_NODE", "1"))
    num_available_workers = num_raycluster_workers * num_gpus_per_raycluster_worker
    num_gpus_per_job_worker = 1

    # ScalingConfig defines distributed set up.
    scaling_config = ScalingConfig(
        num_workers=num_available_workers,
        use_gpu=True,
        resources_per_worker={"GPU": num_gpus_per_job_worker}
    )

    torch_trainer_instance = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=loop_config,
        scaling_config=scaling_config,
    )

    result = torch_trainer_instance.fit()

    print("Fine-tuning job completed.")
    if result.metrics: # Check if metrics exist
        print(f"Results: {result.metrics}")
    else:
        print("No metrics returned from TorchTrainer result.")

    # Use values from loop_config for the final print statement
    sft_output_final_dir = os.path.join(loop_config["OUTPUT_DIR_BASE"], loop_config["SFT_SUBDIR_NAME"])
    print(f"SFTTrainer outputs (model adapters, checkpoints) were saved to: {sft_output_final_dir}")