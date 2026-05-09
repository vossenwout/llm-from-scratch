import math
import torch
from torch.nn import Softmax

a = torch.rand(2, 4, 4)
print(a)
mask = torch.ones_like(a, dtype=torch.bool).triu(diagonal=1)
print(mask)
a = a.masked_fill(mask, float("-inf"))
# a = a.triu()
print(a)
