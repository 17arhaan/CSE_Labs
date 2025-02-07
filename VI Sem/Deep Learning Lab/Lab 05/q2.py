import torch
import torch.nn as nn
import torch.nn.functional as F

image = torch.rand(1, 1, 6, 6)
print("image.shape:", image.shape)

conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, bias=False)
out_conv2d = conv_layer(image)
print("Output from torch.nn.Conv2d:", out_conv2d.shape)

kernel = conv_layer.weight  
out_func_conv2d = F.conv2d(image, kernel, stride=1, padding=0)
print("Output from torch.nn.functional.conv2d:", out_func_conv2d.shape)

print("Are outputs equal?", torch.allclose(out_conv2d, out_func_conv2d))
