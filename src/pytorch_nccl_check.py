import torch
import os
import socket

print(f"--- PyTorch/NCCL Diagnostics ---")
print(f"Hostname: {socket.gethostname()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA Device (from torch): {torch.cuda.current_device()}")
    try:
        cudart_version = torch.version.cuda
        print(f"PyTorch compiled with CUDA Version: {cudart_version}")
    except Exception as e:
        print(f"Could not get PyTorch CUDA version: {e}")
else:
    print("CUDA NOT AVAILABLE according to PyTorch.")

print(f"NCCL Available (PyTorch perspective): {torch.distributed.is_nccl_available()}")

if torch.distributed.is_nccl_available():
    try:
        # This might only work if a process group is initialized,
        # but some PyTorch versions expose it directly.
        nccl_version = torch.cuda.nccl.version() # Tuple: (major, minor, patch)
        print(f"PyTorch detected NCCL Version: {nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}")
    except Exception as e:
        print(f"Could not get PyTorch NCCL version directly via torch.cuda.nccl.version(): {e}")
        print("Attempting to initialize a dummy single-process NCCL group to get version (this might hang or error if NCCL is broken)...")
        # Try to init a single process group to query version if direct query fails
        # This is a common way to check if NCCL can at least load and respond.
        # It does not test multi-GPU or multi-node.
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '29501' # Use a unique port
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                print(f"Dummy NCCL process group initialized. NCCL is likely operational at a basic level.")
                # NCCL_VERSION_CODE is available after init in recent NCCL
                # NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH
                # This is harder to get nicely without parsing nccl.h or running nccl自体
                # The fact that init_process_group doesn't fail is a good sign.
                torch.distributed.destroy_process_group()
            except Exception as e_init:
                print(f"ERROR during dummy NCCL init_process_group: {e_init}")
                print("This suggests a problem with NCCL initialization even for a single process.")
        else:
            print("Skipping dummy NCCL init as no GPU is available to this process.")
else:
    print("NCCL NOT AVAILABLE according to PyTorch.")

print(f"--- End Diagnostics ---")