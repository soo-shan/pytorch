# # Adding dimensions to Tensors
tensor_1 = torch.tensor([1,2,3,4])
print(tensor_1)
print(tensor_1.shape)
tensor_a = torch.unsqueeze(tensor_1,0)
print(tensor_a)
print(tensor_a.shape)

tensor_b = torch.unsqueeze(tensor_1,1)
print(tensor_b)
print(tensor_b.shape)

tensor_2 = torch.rand(2,3,4)
print(tensor_2)
tensor_c = tensor_2[:,:,2]
print(tensor_c)
print(tensor_c.shape)
tensor_d = torch.unsqueeze(tensor_c,2)
print(tensor_d)
print(tensor_d.shape)