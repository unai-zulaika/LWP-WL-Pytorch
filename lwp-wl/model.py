import torch
import torch.nn.functional as F

from layers import Edge_to_Edge_Conv2d, Edge_to_Node_Conv2d

class LWP_WL(torch.nn.Module):
    def __init__(self, K):
        super(LWP_WL, self).__init__()
        #self.conv1_drop = torch.nn.Dropout2d()
        print("LWP_WL")
        self.conv1 = Edge_to_Edge_Conv2d(K)
        # self.conv2 = torch.nn.Conv2d(1, 32, kernel_size=(4,4))
        #self.conv2_drop = torch.nn.Dropout2d()
        #self.conv2 = Edge_to_Node_Conv2d(K)
        self.fc1 = torch.nn.Linear(int(3200), 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.pred = torch.nn.Linear(8, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = self.conv1_drop(x)
        # x = F.relu(self.conv2(x))
        # x = self.conv2_drop(x)
        x = x.view(-1, x.size()[1:].numel())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.pred(x)

        return x

    def reset_model(self):
        for layer in self.children():
            if not isinstance(layer, torch.nn.Dropout2d):
                layer.reset_parameters()

class LWP_WL_SIMPLE_CNN(torch.nn.Module):
    def __init__(self, K=10):
        super(LWP_WL_SIMPLE_CNN, self).__init__()
        print("SIMPLE CNN")
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(4,4))
        # self.conv1_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(int(3136), 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.pred = torch.nn.Linear(8, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        # x = self.conv1_drop(x)
        x = x.view(-1, x.size()[1:].numel())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.pred(x)

        return x

    def reset_model(self):
        for layer in self.children():
            if not isinstance(layer, torch.nn.Dropout2d):
                layer.reset_parameters()

class LWP_WL_NO_CNN(torch.nn.Module):
    def __init__(self, K=10):
        super(LWP_WL_NO_CNN, self).__init__()
        print("NO CNN")
        self.fc1 = torch.nn.Linear(K*K, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.pred = torch.nn.Linear(8, 1)

    def forward(self, x):
        x = x.view(-1, x.size()[1:].numel())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.pred(x)

        return x

    def reset_model(self):
        for layer in self.children():
            if not isinstance(layer, torch.nn.Dropout2d):
                layer.reset_parameters()
