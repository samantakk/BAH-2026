from model import TinyUNet
import torch

model = TinyUNet()
dummy_input = torch.randn(1, 3, 128, 128)  # batch=1, 3 channels, 128x128 image
output = model(dummy_input)
print(output.shape)  # should print: torch.Size([1, 3, 128, 128])