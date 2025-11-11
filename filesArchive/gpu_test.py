import torch

if torch.cuda.is_available():
    print("CUDA доступна!")
    print("Имя GPU:", torch.cuda.get_device_name(0))
    print("Всего памяти GPU (MB):", torch.cuda.get_device_properties(0).total_memory // (1024**2))
else:
    print("CUDA недоступна, будет использоваться CPU")
