import torch

print("="*60)
print("GPU DETECTION TEST")
print("="*60)
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
    
    # Test GPU computation
    x = torch.rand(5, 3).cuda()
    print(f"\n✅ GPU is working! Sample tensor on GPU:")
    print(x)
else:
    print("\n❌ CUDA not available. GPU won't be used.")
