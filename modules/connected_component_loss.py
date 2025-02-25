import torch
from torch_geometric.utils import degree, subgraph


### done checked looking good


def weighted_degree(index, num_nodes, edge_weights):
    """
    Manually compute weighted degree for each node.
    """
    deg = torch.zeros(num_nodes, device=index.device)
    deg.scatter_add_(0, index, edge_weights)
    return deg

def connected_components_loss(node_scores, edge_index, batch, y_true, epsilon=1e-6, reg_lambda=0.01):
    """
    Memory-efficient differentiable connectivity loss using sparse operations.
    """
    num_graphs = batch.max().item() + 1
    loss_values = []
    
    for i in range(num_graphs):
        # Get nodes for current graph
        graph_mask = batch == i
        graph_scores = node_scores[graph_mask]
        
        # Subgraph extraction
        sub_edge_index, _ = subgraph(graph_mask, edge_index, relabel_nodes=True)
        num_nodes = graph_scores.size(0)
        
        # Weight edges by node scores
        edge_weights = graph_scores[sub_edge_index[0]] * graph_scores[sub_edge_index[1]]
        
        # Compute weighted degrees
        deg = weighted_degree(sub_edge_index[0], num_nodes, edge_weights)
        
        # Compute sum of edge weights directly
        edge_weights_sum = edge_weights.sum()
        
        # Compute Laplacian trace efficiently:
        # trace(L) = sum(deg) - sum(edge_weights)
        trace_L = torch.sum(deg) - edge_weights_sum
        
        # Normalize by expected number of functional residues
        norm_factor = torch.clamp(y_true[i], min=epsilon)
        loss_values.append(trace_L / norm_factor)
    
    connectivity_loss = torch.mean(torch.stack(loss_values))
    reg_term = reg_lambda * torch.mean(node_scores ** 2)
    total_loss = connectivity_loss + reg_term
    
    return total_loss, connectivity_loss, reg_term