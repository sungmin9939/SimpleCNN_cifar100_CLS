import torch
import torch.nn as nn



start = torch.arange(1., 5.)
end = torch.empty(4).fill_(10)

output = torch.lerp(start, end, 0.1).unsqueeze(0)

print(output.shape)


