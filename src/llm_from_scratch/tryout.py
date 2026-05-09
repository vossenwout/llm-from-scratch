import torch

a = torch.zeros((4, 8, 3))

a[2, 1, 2] = 9
a[2, 4, 2] = 8
print(a)

print(a[2, :, 2])

a = a[2, :, 2]
print(a)
print(a.argmax())
