import torch
from torch.nn.modules.module import Module


class Edge_to_Edge_Conv2d(Module):
    def __init__(self, K):
        super(Edge_to_Edge_Conv2d, self).__init__()
        # Convolutional Layer columns
        self.conv_column = torch.nn.Conv2d(1, 32, kernel_size=(1, K))
        self.conv_row = torch.nn.Conv2d(1, 32, kernel_size=(K, 1))
        self.K = K

    def forward(self, input):
        conv_column = self.conv_column(input)
        conv_row = self.conv_row(input)
        concat_col_row = torch.cat([conv_row] * self.K, dim=2)
        concat_row_col = torch.cat([conv_column] * self.K, dim=3)
        output = torch.matmul(concat_col_row, concat_row_col)

        return output

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()


class Edge_to_Node_Conv2d(Module):
    def __init__(self, K):
        super(Edge_to_Node_Conv2d, self).__init__()
        # Convolutional Layer columns
        self.conv_column = torch.nn.Conv2d(1, 1, kernel_size=(1, K))
        self.conv_row = torch.nn.Conv2d(1, 1, kernel_size=(K, 1))
        self.K = K

    def forward(self, input):
        conv_column = self.conv_column(input)
        conv_row = self.conv_row(input)
        concat_col_row = torch.cat([conv_row] * self.K, dim=2)
        concat_row_col = torch.cat([conv_column] * self.K, dim=3)
        output = torch.matmul(concat_col_row, concat_row_col)
        print(output.shape)
        output_shaped = torch.ones(size=(output.shape[0], output.shape[1],
                                         self.K),
                                   device=output.device,
                                   dtype=output.dtype)
        for row in range(output.shape[0]):
            output_shaped[row, 0, ...] = torch.diag(output[row, 0])

        return output_shaped

    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()
