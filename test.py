import torch
import platform

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    print("GPU: Apple Silicon MPS ✓")
else:
    print("GPU: None — CPU only")