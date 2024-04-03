import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class FunctionalNodeAttentionGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, device):
        super(FunctionalNodeAttentionGNN, self).__init__(
            aggr="add"
        )  # Use 'add' for aggregating messages.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        # Learnable parameters
        self.W = Parameter(torch.Tensor(in_channels, out_channels))
        self.a = Parameter(
            torch.Tensor(2 * out_channels, 1)
        )  # Attention mechanism parameters

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, edge_index, node_type, batch):
        # x: Node features
        # edge_index: Graph connectivity
        # node_type: Vector indicating node type (e.g., 1 for functional, 0 for contextual)
        # batch: Vector indicating node membership in different graphs within the batch
        # print("func attention x, self.W")
        # print(x.dtype, self.W.dtype)
        x = x.to(torch.float32)
        x = torch.matmul(x, self.W)

        # Prepare for attention computation
        row, col = edge_index
        x_i = x[row]  # Source
        x_j = x[col]  # Target

        # Compute attention coefficients
        alpha = F.leaky_relu(torch.cat([x_i, x_j], dim=1) @ self.a).squeeze(-1)

        # Adjust attention based on node type
        # print(col, node_type.dtype)
        print("node type", node_type)
        node_type = node_type.detach()
        alpha = alpha.detach()  ####### THROWING ERROR WITHOUT THIS LINE?
        alpha = alpha * node_type[col] + alpha * (1 - node_type[col]) * 0.5
        # Apply softmax separately for each graph in the batch
        alpha = alpha.to(self.device)
        alpha = softmax(alpha, col, batch)

        return self.propagate(
            edge_index, size=(x.size(0), x.size(0)), x=x, alpha=alpha, batch=batch
        )

    def message(self, x_j, alpha, index, size_i):
        # Apply the attention-weighted message passing
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        # Update node embeddings
        return aggr_out
