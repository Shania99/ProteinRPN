import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableKHopSubgraph(nn.Module):
    def __init__(self, in_channels, k=2, alpha=0.5):
        """
        Args:
            in_channels (int): Node feature dimension.
            k (int): Number of hops to propagate.
            alpha (float): Propagation factor for soft diffusion.
        """
        super(DifferentiableKHopSubgraph, self).__init__()
        self.k = k
        self.alpha = alpha
        self.scoring_fn = nn.Linear(in_channels, 1)  # Learnable seed node selection

    def forward(self, x, adj, s):
        """
        Args:
            x (Tensor): Node feature matrix (B, N, F).
            adj (Tensor): Dense adjacency matrix (B, N, N).

        Returns:
            x_sub (Tensor): Extracted soft subgraph node features (B, N, F).
            adj_sub (Tensor): Extracted subgraph adjacency matrix (B, N, N).
            selection_scores (Tensor): Soft subgraph selection scores (B, N).
        """
        B, N, _ = x.shape  # Batch size, number of nodes, feature dim

        # Normalize adjacency matrix
        adj = F.normalize(adj, p=1, dim=-1)  # Row-normalize adjacency

        # Compute initial soft selection vector
        # s = torch.sigmoid(self.scoring_fn(x)).squeeze(-1)  # (B, N)

        # Perform k-hop soft propagation
        for _ in range(self.k):
            s = (
                self.alpha * torch.matmul(adj, s.unsqueeze(-1)).squeeze(-1)
                + (1 - self.alpha) * s
            )  # (B, N)

        # Apply soft mask to node features
        x_sub = x * s.unsqueeze(-1)  # (B, N, F)

        # Apply soft mask to adjacency matrix
        adj_sub = s.unsqueeze(1) * adj * s.unsqueeze(2)  # (B, N, N)

        return x_sub, adj_sub, s

    def loss(self, s):
        """Regularization loss to encourage focused subgraph selection."""
        return (-s * torch.log(s + 1e-15) - (1 - s) * torch.log(1 - s + 1e-15)).mean()
