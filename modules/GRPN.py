### change for adj matrix


import torch
from torch.nn import Sigmoid, ReLU
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import DenseGATConv as GATConv
from torch_geometric.nn import DenseGCNConv as GCNConv
from diff_khop_subgraph import DifferentiableKHopSubgraph
import pickle as pkl

from node_drop_pooling import NodeDropPooling
import sys

# sys.append(0, "../HEAL")


class ProjectionLayer(nn.Module):
    def __init__(self):
        super(ProjectionLayer, self).__init__()
        self.one_hot_embed = nn.Embedding(21, 1280)

    def forward(self, esm_embed, native_x):
        esm_embed = esm_embed.float()
        x_aa = self.one_hot_embed(native_x.long())
        return F.relu(esm_embed + x_aa)


class PruningUnit(torch.nn.Module):
    """Pruning unit to process subgraph anchors (ego graphs) and predict nodes to be pruned"""

    def __init__(self, k, input_dim, hidden_dim, output_dim):
        super(PruningUnit, self).__init__()
        self.k = k
        self.conv1 = GATConv(hidden_dim, input_dim, heads=1)
        self.k_hop_subgraph = DifferentiableKHopSubgraph(input_dim, k=k)
        self.pool_layer = NodeDropPooling(input_dim, hidden_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)

        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x, adj, batch, func_bin, ss_tensor=None, coords=None):

        # indices = func_bin.nonzero().squeeze()
        # threshold = torch.max(edge_index).item()
        # filtered_indices = indices[indices <= threshold]

        # func_scores?

        x = F.leaky_relu(self.conv1(x, adj))

        x_sub, adj_sub, _ = self.k_hop_subgraph(x, adj, func_bin)

        x, adj, drop_prob = self.pool_layer(
            x=x_sub,
            adj=adj_sub,
            batch=batch,
            ss=ss_tensor,
            coords=coords,
        )

        return x, adj, drop_prob


class GraphRPN(torch.nn.Module):
    """Graph RPN Model: A GNN model with a pruning unit and a functionality prediction unit"""

    def __init__(
        self, k=2, input_dim=1280, hidden_dim=256, num_classes=1, projection=False
    ):
        super(GraphRPN, self).__init__()
        self.project = projection
        if self.project:
            self.projection_layer = ProjectionLayer()

        self.k_layer_gcn = torch.nn.ModuleList(
            [GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(k)]
        )

        self.graph_pruning_unit = PruningUnit(k, input_dim, hidden_dim, num_classes)

        self.functionality_prediction_unit = GATConv(hidden_dim, num_classes)
        self.sigmoid = Sigmoid()
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x, adj, batch, native_x=None, ss_tensor=None, coords=None):
        x = x.to(torch.float32)
        if self.project:
            x = self.projection_layer(x, native_x)

        for i, gcn in enumerate(self.k_layer_gcn):
            x = gcn(x, adj)
            if i < len(self.k_layer_gcn) - 1:
                x = F.leaky_relu(x)

        functionality_logits = self.functionality_prediction_unit(x, adj)

        func_bin = self.sigmoid((functionality_logits.squeeze() - self.threshold) * 2)

        x, adj, node_drop = self.graph_pruning_unit(
            x, adj, batch, func_bin, ss_tensor=ss_tensor, coords=coords
        )

        return x, adj, node_drop, func_bin
