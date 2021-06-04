import torch
import torch.nn.functional as F

from layers import Edge_to_Edge_Conv2d, Edge_to_Node_Conv2d


class LWP_WL(torch.nn.Module):
    def __init__(self, K):
        super(LWP_WL, self).__init__()
        n_channels = 32
        print("LWP_WL")
        self.conv1 = Edge_to_Edge_Conv2d(K, n_channels)

        self.fc1 = torch.nn.Linear(K*K*n_channels, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.pred = torch.nn.Linear(8, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.flatten(1)
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
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(4, 4))
        self.fc1 = torch.nn.Linear(int(3136), 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 8)
        self.pred = torch.nn.Linear(8, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(
                    m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))

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
        self.fc1 = torch.nn.Linear(K * K, 32)
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

class Node2Vec(torch.nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, weights, dropout):
        super(Node2Vec, self).__init__()
        self.dropout = dropout
        self.W = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, n1_indices, n2_indices):
        n1_val = torch.index_select(self.W, 0, n1_indices)
        n2_val = torch.index_select(self.W, 0, n2_indices)
        
        adj = torch.sum(n1_val * n2_val, dim=1)
        return adj