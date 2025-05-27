import math
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, Checkpoint
from ray.train.torch import TorchTrainer, TorchConfig

import tempfile
import ray.cloudpickle as pickle

class CharTokenizer:
    def __init__(self, vocab_file_path=None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        if vocab_file_path and os.path.exists(vocab_file_path):
            self.load_vocab(vocab_file_path)

    def fit_on_text(self, text):
        chars = sorted(list(set(text)))
        self.char_to_idx = {char: i for i, char in enumerate(chars)}
        self.idx_to_char = {i: char for i, char in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_idx.get(char, -1) for char in text] 

    def decode(self, ids):
        return "".join([self.idx_to_char.get(idx, "") for idx in ids])

    def save_vocab(self, file_path):
        vocab_data = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": self.vocab_size
        }
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        self.char_to_idx = vocab_data["char_to_idx"]
        self.idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
        self.vocab_size = vocab_data["vocab_size"]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0: 
             pe[:, 1::2] = torch.cos(position * div_term[:-1] if div_term.size(0) > 1 else position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class BasicLLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int,
                 hidden_dim: int, max_seq_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len = src.size(1)
        # Generate mask on the same device as src
        src_mask = self._generate_square_subsequent_mask(seq_len, src.device) 
        embedded = self.token_embedding(src) * math.sqrt(self.embed_dim)
        positioned_embedded = self.positional_encoding(embedded)
        final_embedded = self.dropout(positioned_embedded)
        output = self.transformer_decoder(final_embedded, mask=src_mask)
        output = self.fc_out(output)
        return output

class TextDataset(Dataset):
    def __init__(self, token_ids_tensor: torch.Tensor, seq_len: int):
        self.token_ids = token_ids_tensor
        self.seq_len = seq_len
        self.num_sequences = max(0, len(self.token_ids) - self.seq_len -1) # Ensure target is also within bounds

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq = self.token_ids[idx : idx + self.seq_len]
        target_seq = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return input_seq, target_seq


def train_loop_per_worker(config: dict):

    # --- Ray Train Setup ---
    session = train.get_context()
    world_rank = session.get_world_rank()
    world_size = session.get_world_size()
    device = train.torch.get_device()

    # --- Configuration ---
    lr = config["lr"]
    batch_size_per_worker = config["batch_size_per_worker"]
    num_epochs = config["num_epochs"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    hidden_dim = config["hidden_dim"]
    model_max_seq_len = config["model_max_seq_len"]
    dataset_seq_len = config["dataset_seq_len"]
    dataloader_num_workers = config.get("dataloader_num_workers", 0)

    raw_data_path = config["raw_data_path"]
    processed_data_dir = config["processed_data_dir"]
    
    tokenized_data_file = os.path.join(processed_data_dir, "train.ids.pt")
    vocab_file = os.path.join(processed_data_dir, "char_vocab.json")
    vocab_size_file = os.path.join(processed_data_dir, "vocab_size.txt")
    data_prep_done_file = os.path.join(processed_data_dir, "_DATA_PREP_DONE")
    test_run = config.get("test_run", True)

    storage_path_base_on_fuse = config["storage_path_base_on_fuse"]
    trial_name = session.get_trial_name()

    actual_vocab_size = 0

    if world_rank == 0:
        print(f"Worker Rank {world_rank}/{world_size} on device {device}. CUDA: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
        os.makedirs(processed_data_dir, exist_ok=True)
        
        if not os.path.exists(data_prep_done_file):
            print(f"Rank 0: Data prep files not found or incomplete. Starting/Re-doing data preparation...")
            tokenizer = CharTokenizer()
            try:
                with open(raw_data_path, 'r', encoding='utf-8') as f: text_content = f.read()
                print(f"Rank 0: Read {len(text_content):,} characters from {raw_data_path}.")
            except Exception as e:
                print(f"Rank 0: ERROR reading raw data file {raw_data_path}: {e}")
                raise
            
            tokenizer.fit_on_text(text_content)
            tokenizer.save_vocab(vocab_file)
            token_ids = tokenizer.encode(text_content)
            torch.save(torch.tensor(token_ids, dtype=torch.long), tokenized_data_file)
            actual_vocab_size = tokenizer.vocab_size
            with open(vocab_size_file, 'w') as f: f.write(str(actual_vocab_size))
            with open(data_prep_done_file, 'w') as f: f.write("done") # Create done flag
            print(f"Rank 0: Data prepared. Vocab size: {actual_vocab_size}. Tokens: {len(token_ids)}. Saved to GCS Fuse.")
        else:
            print(f"Rank 0: Found existing data prep done flag. Loading vocab size.")
            with open(vocab_size_file, 'r') as f: actual_vocab_size = int(f.read().strip())
            print(f"Rank 0: Loaded existing vocab size: {actual_vocab_size}")
    else:
        print(f"Rank {world_rank}: Waiting for data preparation by Rank 0...")
        while not os.path.exists(data_prep_done_file):
            time.sleep(5)
        print(f"Rank {world_rank}: Data preparation done signal received.")
        with open(vocab_size_file, 'r') as f: actual_vocab_size = int(f.read().strip())
        print(f"Rank {world_rank}: Loaded vocab size: {actual_vocab_size}")

    # All workers load the tokenized data and vocab
    all_token_ids = torch.load(tokenized_data_file)
    if world_rank == 0: print(f"All ranks proceeding. Vocab size: {actual_vocab_size}. Loaded {len(all_token_ids)} tokens.")

    # Load dataset
    train_dataset = TextDataset(token_ids_tensor=all_token_ids, seq_len=dataset_seq_len)

    # Sub sample dataset if specified
    if test_run:
        num_samples_to_select = 16000
        indices = list(range(min(num_samples_to_select, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if len(train_dataset) == 0:
        raise ValueError(f"Rank {world_rank}: Dataset empty after loading. Tokens: {len(all_token_ids)}, seq_len: {dataset_seq_len}")

    pytorch_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size_per_worker,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False, 
        num_workers=dataloader_num_workers,
        drop_last=True 
    )

    # Use Ray Train to prepare the DataLoader for distributed training
    train_dataloader = train.torch.prepare_data_loader(pytorch_dataloader)
    
    # Calculate num_batches_per_epoch based on the length of the *prepared* DataLoader for this rank
    num_batches_per_epoch = len(train_dataloader) 
    if world_rank == 0: 
        print(f"Ray Train prepared DataLoader. Batches per epoch on this worker (Rank 0): {num_batches_per_epoch}")


    # --- Model setup ---
    model = BasicLLM(
        vocab_size=actual_vocab_size, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, hidden_dim=hidden_dim, max_seq_len=model_max_seq_len
    )    
    # Use Ray Train to prepare the model (moves to device, wraps in DDP)
    model = train.torch.prepare_model(model)
    
    if world_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model prepared by Ray Train. Trainable params (Rank 0): {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # --- Learning Rate Scheduler Setup ---
    total_training_steps = num_epochs * num_batches_per_epoch
    warmup_steps_ratio = config.get("warmup_steps_ratio", 0.05)
    min_lr_ratio = config.get("min_lr_ratio", 0.01)
    num_warmup_steps = int(total_training_steps * warmup_steps_ratio)
    num_decay_steps = total_training_steps - num_warmup_steps

    if world_rank == 0:
        print(f"LR Scheduler: Total steps (per worker): {total_training_steps}, Warmup: {num_warmup_steps}, Decay: {num_decay_steps}")

    def lr_lambda(current_step):
        # ... (LR lambda function as before) ...
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < total_training_steps: # Use total_training_steps here
            progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        return min_lr_ratio
    scheduler = LambdaLR(optimizer, lr_lambda)

    # --- Training loop ---
    current_global_step = 0 # This will track steps across epochs for this worker

    for epoch in range(num_epochs):

        if hasattr(train_dataloader, 'sampler') and isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        
        if world_rank == 0: print(f"Rank 0: Epoch {epoch+1} starting.")
        
        for batch_idx, batch_data in enumerate(train_dataloader): # Iterate over the Ray-prepared DataLoader
            
            inputs, targets = batch_data # Data is already on device via prepare_data
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs) # Model is already on device via prepare_model
            loss = criterion(outputs.reshape(-1, actual_vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            current_global_step += 1
            
            if world_rank == 0 and (batch_idx % config.get("log_frequency_batches", 100) == 0 or batch_idx == num_batches_per_epoch - 1):
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches_per_epoch}, Step {current_global_step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        # --- End of Epoch Reporting ---
        epoch_summary_metrics = {
            "loss": loss.item(), # Use last recorded loss
            "epoch": epoch + 1,
            "learning_rate_epoch_end": scheduler.get_last_lr()[0],
            "global_step_epoch_end": current_global_step
        }

        # Epoch checkpoint - only rank 0 because DDP.
        # # https://docs.ray.io/en/latest/train/user-guides/checkpoints.html
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:

            checkpoint = None # Initialize for all workers

            if world_rank == 0:        
                torch.save(model.module.state_dict(), os.path.join(temp_checkpoint_dir, "model.pth"))
                torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pth"))
                torch.save(scheduler.state_dict(), os.path.join(temp_checkpoint_dir, "scheduler.pth"))

                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
                            
                print(f"Epoch {epoch+1} finished. Last Batch Loss: {loss.item():.4f}.")
    
            # All workers must call train.report() at end of epoch.
            train.report(metrics=epoch_summary_metrics, checkpoint=checkpoint)
    
    if world_rank == 0:
        print(f"Rank {world_rank} training loop finished.")

# --- 5. Main Function to Configure and Run Trainer (Adjusted) ---
if __name__ == "__main__":
    N_TOTAL_GPUS = 16
    effective_batch_size_per_gpu = 16

    experiment_name = "wikitext2_manualTB_v1" # New name for this run
    storage_path_base = "/mnt/pvc/ray_llm_training_runs" # Base GCS FUSE path for all runs


    train_loop_config = {
        "lr": 3e-4,
        "batch_size_per_worker": effective_batch_size_per_gpu,
        "num_epochs": 1, # Keep it at 5 for a decent run
        "embed_dim": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "hidden_dim": 2048 * 4,
        "model_max_seq_len": 1024,
        "dataset_seq_len": 256,
        "dataloader_num_workers": 0, # Start with 0 for stability, can increase later
        "log_frequency_batches": 20,
        "train_report_frequency_steps": 20, # Report to TB more often
        "warmup_steps_ratio": 0.05,
        "min_lr_ratio": 0.01,
        "raw_data_path": "/mnt/pvc/datasets/wikitext-2-raw/wiki.train.tokens",
        "processed_data_dir": "/mnt/pvc/datasets/wikitext-2-processed/",
        "storage_path_base_on_fuse": storage_path_base,
        "experiment_name_for_tb": experiment_name,
        "test_run": True
    }

    scaling_config = ScalingConfig(
        num_workers=N_TOTAL_GPUS,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )
    
    run_config = RunConfig(
        name=experiment_name,
        storage_path=storage_path_base,
        checkpoint_config=train.CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min"
        ),
    )

    torch_config = TorchConfig(
        backend="nccl",
    )

    print(f"Starting Ray Train job with Ray utilities: {experiment_name}")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        torch_config=torch_config # Pass TorchConfig
    )

    result = trainer.fit()

    print(f"\n--- Training Finished for {experiment_name} ---")