import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os

class MeinNetz(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

netz = MeinNetz()

# if os.path.isfile('meinNetz.pt'):
#     netz = torch.load('meinNetz.pt')

# learning Loop vom Netz, passiert hier in Batches Größe 10 
# das heißt es wird 10 mal Input mit 10 mal korrekten Output verglichen und dann optmiert
# das ganze halt 100 mal
# lr im optimizer ist  die learning rate
for i in range(100):
    input = Variable(torch.randn(10, 4))
    # print(input)
    out = netz(input)
    # print(out)

    x = [1, 0]
    target = Variable(torch.Tensor([x for _ in range(10)]))
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    netz.zero_grad()
    loss.backward()
    optimizer = optim.SGD(netz.parameters(), lr= 0.2 )
    optimizer.step()

input = Variable(torch.Tensor([1, 2, 3, 4]))
print(netz(input))
 
# torch.save(netz, 'meinNetz.pt')