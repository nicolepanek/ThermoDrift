import torch

x = torch.rand(10)

for i in range(x.size()[0]):
	print(x[i])

