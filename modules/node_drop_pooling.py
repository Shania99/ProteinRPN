import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv


class NodeDropPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(NodeDropPooling, self).__init__()

        # GNN to compute drop probabilities
        self.gnn_drop = DenseGCNConv(in_channels, hidden_channels)

        # Projection layer to obtain a single probability per node
        self.prob_proj = nn.Linear(hidden_channels, 1)

        self.sigmoid = nn.Sigmoid()
        self.entropy_reg_weight = 1e-3  # Regularization weight for entropy loss

    def forward(self, x, adj, batch, ss, coords):
        """
        Args:
            x (Tensor): Node features of shape [batch_size, num_nodes, in_channels]
            adj (Tensor): Dense adjacency matrix [batch_size, num_nodes, num_nodes]
        Returns:
            Updated node features, updated adjacency, and entropy regularization loss
        """
        # Compute drop probability matrix D(l)
        h = self.gnn_drop(x, adj)  # Shape: [batch_size, num_nodes, hidden_channels]
        D = self.sigmoid(self.prob_proj(h))  # Shape: [batch_size, num_nodes, 1]

        # Update node features (element-wise multiplication)
        x = x * D  # Shape remains [batch_size, num_nodes, in_channels]

        # Update adjacency matrix
        adj = (
            D @ D.transpose(1, 2)
        ) * adj  # Shape remains [batch_size, num_nodes, num_nodes]

        # Regularization for differentiability (Entropy loss)
        # entropy_loss = -torch.mean(D * torch.log(D + 1e-8) + (1 - D) * torch.log(1 - D + 1e-8))

        return x, adj, D.detach()
