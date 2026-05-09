import math
import torch

SEQ_L = 8
DIMENSION = 16

pe = torch.zeros(SEQ_L, DIMENSION)
for pos in range(SEQ_L):
    for i in range(DIMENSION):
        if i % 2 == 0:
            pe[pos, i] = math.sin(i / (10000 ** ((2 * i) / DIMENSION)))
        else:
            pe[pos, i] = math.cos(i / (10000 ** ((2 * i) / DIMENSION)))

print(pe)

res = math.sin(3.14 / 2)
print(res)
