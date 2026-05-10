import math
import torch
from torch.nn import Softmax, LayerNorm, ReLU

relu = ReLU()
a = torch.rand(2, 4, 4)
a *= -1
a = relu(a)

print(a)
