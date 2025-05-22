import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import ray.train.torch
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import os
import time
import socket # For printing hostname

# A very simple model
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# This is the function that will be executed by each Ray Train worker (DDP process)
def train_func(config): # Ray Train passes a config dict
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    hostname = socket.gethostname()

    # Set the CUDA device for this DDP process
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): Using CUDA device: {device}")

    # Create a simple model and move it to the GPU
    model = ToyModel().to(device)

    # Wrap the model with DistributedDataParallel (DDP)
    print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): Wrapping model with DDP...")
    try:
        ddp_model = DDP(model, device_ids=[local_rank])
        print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): Model wrapped with DDP successfully.")
    except Exception as e:
        print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): ERROR during DDP model wrapping: {e}")
        raise

    # Perform a simple all-reduce operation to test communication
    tensor_to_reduce = torch.ones(10, device=device) * (rank + 1) # Each rank contributes its rank+1
    print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): Tensor before all_reduce: {tensor_to_reduce.cpu().numpy().mean():.2f} (all elements)")

    try:
        dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.SUM)
        print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): All_reduce completed.")
        
        # Validation: Sum of 1 to N is N*(N+1)/2. Each element was (rank+1).
        # So each element in the sum tensor should be sum_{i=0}^{world_size-1} (i+1) = sum_{j=1}^{world_size} j
        expected_sum_val = float(world_size * (world_size + 1) / 2)
        if torch.allclose(tensor_to_reduce, torch.full_like(tensor_to_reduce, expected_sum_val)):
            print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): All_reduce validation SUCCESSFUL. Result: {tensor_to_reduce.cpu().numpy().mean():.2f}")
        else:
            print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): All_reduce validation FAILED. Expected all elements {expected_sum_val}, Got: {tensor_to_reduce.cpu().numpy()}")
    except Exception as e:
        print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): ERROR during all_reduce: {e}")
        raise

    # Perform a barrier to synchronize all processes
    try:
        print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): Entering barrier...")
        dist.barrier()
        print(f"Host: {hostname}, Rank {rank}/{world_L_SIZE} (Local Rank {local_rank}): Exited barrier successfully.")
    except Exception as e:
        print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): ERROR during barrier: {e}")
        raise

    dist.destroy_process_group()
    print(f"Host: {hostname}, Rank {rank}/{world_size} (Local Rank {local_rank}): Test completed successfully for this rank.")
    ray.train.report({"status": "success", "rank": rank, "hostname": hostname})
    time.sleep(2) # Keep pod alive for a moment more to ensure logs flush


if __name__ == "__main__":
    # Configuration for Ray Train
    # You have 2 nodes, 8 GPUs per node. Total 16 GPUs/ranks.
    # Ray Train's `num_workers` here means the number of DDP processes.
    num_ddp_processes = 16 
    use_gpu_per_worker = True

    scaling_config = ScalingConfig(
        num_workers=num_ddp_processes, 
        use_gpu=use_gpu_per_worker,
        # resources_per_worker={"GPU": 1} # This is implied by use_gpu=True and num_workers
    )

    # Define where to store results (even for a simple test)
    # Ensure /mnt/pvc is writable from your Ray worker pods
    storage_path = "/mnt/pvc/ray_results_simple_ddp_test"
    experiment_name = "SimpleDDPTest"
    
    print(f"Attempting to create storage path: {storage_path}")
    # No easy way to mkdir from job submission script's driver side
    # The worker side will handle this if using Ray AIR persistent storage.
    # For this test, Ray AIR may not even need to write much if train_func doesn't call report() often.

    run_config = RunConfig(
        name=experiment_name,
        storage_path=storage_path,
        checkpoint_config=CheckpointConfig(num_to_keep=1), # Keep at least one checkpoint
        verbose=2 # Print more logs from Ray Train driver
    )

    # Create the TorchTrainer
    # The train_func will be run by `num_ddp_processes` workers.
    # Each worker will be one DDP rank.
    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={}, # Pass any config to train_func if needed
        scaling_config=scaling_config,
        run_config=run_config
    )

    print(f"Submitting TorchTrainer job for {experiment_name}...")
    try:
        result = trainer.fit()
        print(f"TorchTrainer job '{experiment_name}' finished.")
        print(f"Last result: {result.metrics}")
        print(f"Path to results: {result.path}")
    except Exception as e:
        print(f"TorchTrainer job '{experiment_name}' FAILED.")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()