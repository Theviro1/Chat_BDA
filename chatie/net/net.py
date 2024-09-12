import torch

class CustomNet(torch.nn.Module):
    def __init__(self, input_dim:int, num_classes:int):
        super(CustomNet, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, 16)
        self.layer2 = torch.nn.Linear(16, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.layer2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x
