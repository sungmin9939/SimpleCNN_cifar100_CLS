import torch


a = torch.arange(1., 26.).reshape(5,5)
print(a)
print(torch.sum(a, dim=1)/5)


'''
start = torch.arange(1., 5.)
end = torch.empty(4).fill_(10)

print(torch.lerp(start, end, 0.1))
'''

