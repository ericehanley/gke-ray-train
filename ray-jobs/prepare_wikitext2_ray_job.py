import ray
import os
import shutil # Still useful for directory checks if needed, though less critical now
from datasets import load_dataset # Requires 'datasets' library
import time

# --- Configuration ---
# Target base path on the GCS FUSE mount within the Ray pods.
# This corresponds to gs://eh-ray/datasets/wikitext-2-raw/ in your GCS bucket.
GCS_FUSE_TARGET_BASE_PATH = "/mnt/pvc/datasets/wikitext-2-raw"

# Define target file names consistently with what the training script expects
TARGET_TRAIN_FILE = os.path.join(GCS_FUSE_TARGET_BASE_PATH, "wiki.train.tokens")
TARGET_VALID_FILE = os.path.join(GCS_FUSE_TARGET_BASE_PATH, "wiki.valid.tokens")
TARGET_TEST_FILE  = os.path.join(GCS_FUSE_TARGET_BASE_PATH, "wiki.test.tokens")


@ray.remote(num_cpus=1) # Dataset loading can use some CPU for processing
def prepare_wikitext2_via_hf_library():
    """
    A Ray task to download WikiText-2 using the Hugging Face datasets library
    and save the raw text content to the GCS FUSE mount point.
    """
    print(f"Ray Task: Starting WikiText-2 data preparation using Hugging Face 'datasets' library...")
    task_start_time = time.time()

    try:
        # 1. Ensure the target directory on GCS FUSE mount exists
        print(f"Ray Task: Ensuring target GCS FUSE directory exists: {GCS_FUSE_TARGET_BASE_PATH}")
        os.makedirs(GCS_FUSE_TARGET_BASE_PATH, exist_ok=True)

        splits_to_process = {
            "train": TARGET_TRAIN_FILE,
            "validation": TARGET_VALID_FILE,
            "test": TARGET_TEST_FILE,
        }

        # Check if all target files already exist and are non-empty to potentially skip
        all_files_exist_and_are_valid = True
        for target_file_path in splits_to_process.values():
            if not os.path.exists(target_file_path) or os.path.getsize(target_file_path) == 0:
                all_files_exist_and_are_valid = False
                break
        
        if all_files_exist_and_are_valid:
            print(f"Ray Task: All target text files already exist and are non-empty in {GCS_FUSE_TARGET_BASE_PATH}. Skipping download and processing.")
            return f"Data already exists at {GCS_FUSE_TARGET_BASE_PATH}"

        # If any file is missing or empty, process all of them
        print(f"Ray Task: One or more target files missing or empty. Processing all splits...")
        for split_name, target_file_path in splits_to_process.items():
            print(f"Ray Task: Loading '{split_name}' split of 'wikitext' dataset, config 'wikitext-2-raw-v1'...")
            # The `datasets` library will download and cache the data.
            # Default cache is typically ~/.cache/huggingface/datasets in the container.
            # For wikitext-2-raw-v1, each item in dataset['text'] is a string (often a paragraph).
            dataset = load_dataset(
                "wikitext", 
                "wikitext-2-raw-v1", 
                split=split_name, 
                # trust_remote_code=True # May be needed for some datasets; try without first
            )
            
            print(f"Ray Task: Processing and concatenating text for '{split_name}' split (found {len(dataset)} paragraphs/entries)...")
            # The 'wikitext-2-raw-v1' config yields each paragraph/entry as a separate string in dataset['text'].
            # We will join them with newlines to make one large text block per split,
            # which is suitable for our character tokenizer.
            full_text = "\n".join(dataset['text'])

            print(f"Ray Task: Writing concatenated text for '{split_name}' split to {target_file_path}...")
            with open(target_file_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            file_size_mb = os.path.getsize(target_file_path) / (1024*1024)
            print(f"Ray Task: Successfully wrote '{split_name}' data to {target_file_path}. Size: {file_size_mb:.2f} MB")

        task_end_time = time.time()
        duration = task_end_time - task_start_time
        print(f"Ray Task: Data preparation finished successfully in {duration:.2f} seconds.")
        print("---------------------------------------------------------------------")
        print("The following raw text files should now be available on GCS via the FUSE mount:")
        for split_name, target_file_path in splits_to_process.items():
            file_size_mb = os.path.getsize(target_file_path) / (1024*1024) if os.path.exists(target_file_path) else 0
            print(f"  {split_name.capitalize()}: {target_file_path} ({file_size_mb:.2f} MB)")
        print("---------------------------------------------------------------------")
        return f"Successfully prepared data at {GCS_FUSE_TARGET_BASE_PATH} using HF datasets library."

    except Exception as e:
        print(f"Ray Task: An error occurred during data preparation: {type(e).__name__} - {e}")
        import traceback
        print("Ray Task: Traceback follows:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    if not ray.is_initialized():
        print("Ray not initialized. Attempting to connect to Ray cluster with address='auto'...")
        ray.init(address='auto', ignore_reinit_error=True) 
        print("Ray connection attempt complete.")

    print("Submitting WikiText-2 data preparation task (using Hugging Face 'datasets' library) to the Ray cluster...")
    
    # Define runtime environment if 'datasets' or its dependencies might be missing,
    # though ray-ml images usually include it. For explicit versioning or ensuring presence:
    # runtime_env = {"pip": ["datasets==2.15.0"]} # Example version, adjust as needed
    # data_prep_task_ref = prepare_wikitext2_via_hf_library.options(runtime_env=runtime_env).remote()
    data_prep_task_ref = prepare_wikitext2_via_hf_library.remote()

    try:
        result_message = ray.get(data_prep_task_ref, timeout=1800) # 30 minute timeout
        print(f"\nRay Job Main: Task completed. Result: {result_message}")
    except Exception as e:
        print(f"\nRay Job Main: Data preparation task failed or timed out.")
        print(f"Error details: {type(e).__name__} - {e}")

    print("Ray Job Main: Script finished.")