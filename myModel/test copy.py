import torch
import torch.nn.functional as F
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1) #(Cin, Cout, Ksiize, stride)->(26,26,32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)          # ->(24,24,64)
        self.dropout1 = torch.nn.Dropout(0.25)              # ->()
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)  
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x)
        x = self.dropout1(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class ModelWrapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = Net()
    def forward(self,x):
        return self.mod(x)


def get_model():
    model = ModelWrapper()
    return model

aaa = get_model()

print(aaa)


batch_norm = True

bias=False if batch_norm else True
print(bias)