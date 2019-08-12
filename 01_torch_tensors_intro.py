import torch
import torchvision
# print(torch.cuda.is_available())

# # Torch Tensors

# 1D tensor
a = torch.tensor([2,2,1])
print(a)

# 2D tensor
b = torch.tensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]])
print(b)

# size of tensors
print(a.shape)
print(b.shape)
print(a.size())
print(b.size())
# shape is an alias for size

# get the height/number of rows of b
print(b.shape[0])

# float tensor
c = torch.FloatTensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]])
print(c)
print(c.dtype)
c2 = torch.tensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]],dtype=torch.double)
print(c2)
print(c2.dtype)
# Double tensor
d = torch.DoubleTensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]])
print(d)
print(d.dtype)
d2 = torch.tensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]],dtype=torch.double)
print(d2)
print(d2.dtype)

# mean and standard deviation
print(c.mean())
print(d.std())

# reshaping tensors
# Note: If one of the dimensions is -1, its size can be inferred
print(b.view(-1,1))
print(b.view(12))
print(b.view(-1,4))
print(b.view(3,4))
# assign b a new shape
b = b.view(1,-1)
print(b)
print(b.shape)
# reshape 3D tensors
# 3D tensors in pytorch: (channels,rows,columns)
three_dim = torch.randn(2,3,4)
print(three_dim)
print(three_dim.view(2,12))
print(three_dim.view(2,-1))

# create a matrix with random numbers btw 0 & 1
r = torch.rand(4,4)
print(r)
print(r.dtype)
# create a matrix with random numbrs taken from a normal 
# distribution with mean 0 and std dev 1
r2 = torch.randn(4,4)
print(r2)
print(r2.dtype)

# create an array of 5 random integers from value btw 6 & 9
in_array = torch.randint(6,10,(5,)) # low,high, shape
print(in_array)
print(in_array.dtype)

# create a 2-D array(matrix) of size 3x3 with random integers
#  between 6 & 9
in_array2d = torch.randint(6,10,(3,3))
print(in_array2d)

# number of elements in array
print(torch.numel(in_array))
print(torch.numel(in_array2d))

# tensor of zeros
z = torch.zeros(3,3,3, dtype=torch.long) # default float32
print(z)
print(z.dtype)
# tensor of ones
o = torch.ones(3,2,5,2)
print(o)
print(o.dtype)

# copy size with random numbers
r2_like = torch.randn_like(r2,dtype=torch.double)
print(r2)
print(r2_like)
print(r2.shape)
print(r2_like.shape)

# add two tensors
# check shapes
print(r.shape)
print(r2.shape)
add_result = torch.add(r,r2)
print(add_result)

# inplace addition
# note the underscore add_
r2.add_(r) # equivalent to r2 += r
print(r2)

# slicing
print(r2[:-1,:])
print(r2[1,:])
