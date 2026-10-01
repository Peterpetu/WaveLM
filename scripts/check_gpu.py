import torch
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"  Total: {total/1024**3:.2f} GB")
print(f"  Free:  {free/1024**3:.2f} GB")
print(f"  Used:  {(total-free)/1024**3:.2f} GB (by all processes on the device)")
print(f"  PyTorch allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"  PyTorch reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
