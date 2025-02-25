import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, dense_to_sparse


# def map_psi(x, r):
#     """Maps node features to a transformed space where attention is computed"""
#     x_x = x[..., :-1]  # Extract all but last dimension
#     x_y = F.sigmoid(x[..., -1])  # Apply sigmoid to last dimension
#     return x_x * x_y.unsqueeze(-1) * r, x_y * r


# def penumbral(q, k, r=1, gamma=1, eps=1e-6):
#     """Computes function-aware geometric attention for node pairs"""
#     q_x, q_y = map_psi(q, r)
#     k_x, k_y = map_psi(k, r)

#     q_y = q_y.unsqueeze(-1)
#     k_y = k_y.unsqueeze(-1)

#     x_q_y = torch.sqrt(r**2 - q_y**2 + eps)
#     x_k_y = torch.sqrt(r**2 - k_y**2 + eps)

#     pairwise_dist = torch.cdist(q_x, k_x)

#     lca_height = torch.maximum(
#         torch.maximum(q_y**2, k_y**2), r**2 - ((x_q_y + x_k_y - pairwise_dist) / 2) ** 2
#     )

#     lca_height_outcone = (
#         (pairwise_dist**2 + k_y**2 - q_y**2) / (2 * pairwise_dist + eps)
#     ) ** 2 + q_y**2

#     exists_cone = torch.logical_or(
#         pairwise_dist <= x_q_y, (pairwise_dist - x_q_y) ** 2 + k_y**2 <= r**2
#     )

#     return -gamma * torch.where(exists_cone, lca_height, lca_height_outcone)


class FunctionalNodeAttentionGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(FunctionalNodeAttentionGNN, self).__init__(
            aggr="add"
        )  # Sum-based message passing
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable transformation weights
        self.W = Parameter(torch.Tensor(in_channels, out_channels))
        self.a = Parameter(
            torch.Tensor(2 * out_channels, 1)
        )  # Standard attention weight

        self.leaky_relu = nn.LeakyReLU(0.2)  # LeakyReLU for attention scores

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable parameters."""
        torch.nn.init.xavier_uniform_(self.W, gain=1.414)
        torch.nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, x, edge_index, edge_weight, func_node, batch):
        """
        Args:
            x (Tensor): Node feature matrix (N, F).
            edge_index (Tensor): Graph connectivity (2, E).
            edge_weight (Tensor): Edge weights (E,).
            func_node (Tensor): Functionality scores (N,). Continuous values in [0,1].
            batch (Tensor): Batch indices for multiple graphs.

        Returns:
            Tensor: Updated node representations (N, F).
        """

        # Transform node features
        x = torch.matmul(x, self.W)  # (N, F')

        # Extract source & target nodes
        row, col = edge_index
        x_i = x[row]  # Source node features (E, F')
        x_j = x[col]  # Target node features (E, F')

        # Compute raw attention scores
        alpha = self.leaky_relu(torch.cat([x_i, x_j], dim=1) @ self.a).squeeze(-1)  # E

        # Apply geometric function-aware attention (Penumbral)
        # penumbral_score = penumbral(x, x) # N, N
        # print("penumbral_score.shape", penumbral_score.shape)
        # edge_index, penumbral_edge_attn = dense_to_sparse(penumbral_score)

        # # Combine both attention mechanisms
        # alpha = alpha + penumbral_edge_attn

        # Scale attention by functional importance of receiving node
        alpha = alpha * (func_node[col] + 0.5 * (1 - func_node[col]))

        # Scale attention using edge weights
        alpha = alpha * edge_weight  # Weight edges based on structural strength

        # Apply softmax for proper normalization
        alpha = softmax(alpha, col)  # Normalize per node

        # Apply message passing
        return self.propagate(edge_index, x=x, alpha=alpha)

    def message(self, x_j, alpha):
        """
        Compute message for each edge.
        x_j: Target node features (E, F).
        alpha: Attention coefficients (E,).
        """
        return x_j * alpha.unsqueeze(-1)  # (E, F) * (E, 1) → (E, F)

    def update(self, aggr_out):
        """Update node embeddings after aggregation."""
        return F.relu(aggr_out)  # Apply non-linearity
