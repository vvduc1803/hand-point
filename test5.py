
import torch

scale = torch.rand((8, 1, 32, 32))
weight = torch.rand((1, 1, 32, 32))
depth = torch.rand((8, 32, 32))
img = torch.rand((8, 256, 32, 32))
# print(depth)
g = torch.cat((weight * scale, (1 - weight) * torch.unsqueeze(depth, dim=1)), dim=1)
print(g.shape)
scale = torch.sum(g, dim=1, keepdim=True)
# print(scale)
print(img.shape)
print(scale.shape)
a = img*scale
print(a.shape)