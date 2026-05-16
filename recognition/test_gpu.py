import torch
print(f"PyTorch 是否可用 GPU: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前显卡: {torch.cuda.get_device_name(0)}")