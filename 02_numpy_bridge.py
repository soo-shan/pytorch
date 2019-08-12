import numpy as np
import torch

# converting a torch tensor into numpy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b) # value of b also gets modified

# converting numpy array to torch tensor
a = np.zeros(5)
print(a)
b = torch.from_numpy(a)
print(b)
np.add(a,1,out=a)
print(a)
print(b) # value of b also gets modified

# moving the tensor to the GPU
r2 = torch.randn(4,4)
# r2 = r2.cuda()
# print(r2)

#Provide Easy switching between CPU and GPU
r = torch.rand(4,4)
print(r)

CUDA = torch.cuda.is_available()
print(CUDA)
if CUDA:
    r = r.cuda()
    print(r)

# convert list into tensor
a = [2,5,5,8,1]
print(a)
to_list = torch.tensor(a)
print(to_list, to_list.dtype)

data =  [[1., 2.], [3., 4.],[5., 6.], [7., 8.]]
T = torch.tensor(data)
print(T, T.dtype)

