import torch
import platform

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} ✓")
elif torch.backends.mps.is_available():
    print("GPU: Apple Silicon MPS ✓")
else:
    print("GPU: None — CPU only")