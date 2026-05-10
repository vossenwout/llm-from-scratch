import math
import torch
from torch.nn import Softmax

a = torch.rand(2, 4, 4)
print(a)
print(a.shape)
print("---Operation---")
a = torch.cat(
    tensors=[a, a, a],
    dim=2,
)
print(a)
print(a.shape)
