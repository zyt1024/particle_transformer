import torch.nn as nn
import torch

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            print("----", i)
            print(l)
            # print(self.linears[i])
            # print(l(x))
            x = l(x)
        return x

model = MyModule()

print(model)
x = torch.rand(10, 10)
model(x)
