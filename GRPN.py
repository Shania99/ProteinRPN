import torch
from torch.nn import Sigmoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import conv

# from torch_geometric.nn import SAGPooling
from node_drop_pooling import CustomNodeDropPoolingLayer
from torch_geometric.utils import k_hop_subgraph

# class GCNLayer(torch.nn.Module):
#     """A simple GCN layer

#     Returns:
#         torch (Tensor): Output feature tensor
#     """
#     def __init__(self, input_dim, output_dim):
#         super(GCNLayer, self).__init__()
#         self.conv = GCNConv(input_dim, output_dim)

#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         x = F.relu(x)
#         return x


# class GATLayer(torch.nn.Module):
#     """A simple GAT layer

#     Returns:
#         torch (Tensor): Output feature tensor
#     """
#     def __init__(self, input_dim, output_dim):
#         super(GATLayer, self).__init__()
#         self.conv = GATConv(input_dim, output_dim, heads=3)

#     def forward(self, x, edge_index):
#         x = self.conv(x, edge_index)
#         return x


class PruningUnit(torch.nn.Module):
    """Pruning unit to process subgraph anchors (ego graphs) and predict nodes to be pruned

    Returns:
        torch (List[Tensors], List[Tensors]): list of node score tensors for each ego graph, and nodes as part of ego graph for one batch
    """

    def __init__(self, k, input_dim, hidden_dim, output_dim):
        """_summary_
        hidden_dim = 128 for example
        input_dim = 1280 (esm_embedding size)
        output_dim = 1 (for functional prediction)
        Args:
            k (_type_): _description_
            input_dim (_type_): _description_
            output_dim (_type_): _description_
        """
        super(PruningUnit, self).__init__()
        self.k = k
        self.conv1 = GATConv(hidden_dim, input_dim, heads=1)
        # self.conv2 = GATConv(input_dim, output_dim, heads=1)
        # self.pool_layer = SAGPooling(
        #     output_dim,
        #     ratio=0.99,
        #     GNN=GATConv,
        #     nonlinearity="tanh",
        #     multiplier=10,
        #     # nonlinearity=torch.nn.LeakyReLU(0.1),
        # )
        # self.pool_layer = CustomSAGPooling(
        #     output_dim,
        #     ratio=0.99,
        #     GNN=GATConv,
        #     nonlinearity="tanh",
        #     multiplier=10,
        #     # nonlinearity=torch.nn.LeakyReLU(0.1),
        # )

        self.pool_layer = CustomNodeDropPoolingLayer(
            input_dim,
            GNN=GATConv,
            # nonlinearity=torch.nn.LeakyReLU(0.1),
        )
        # self.pool_layer = Custom
        # self.linear = torch.nn.Linear(output_dim, output_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.sigmoid = Sigmoid()

    def forward(self, x, x_orig, edge_index, batch):
        """features used are original esm embeddings of each of the

        Args:
            x (_type_): _description_
            edge_index (_type_): _description_
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.conv1(
            x, edge_index
        )  # INCREasing dimension from hidden dim to input dim
        subgraph_list = []
        subgraph_batch_list = []
        ego_nodes = []
        # Iterate over all unique graphs in the batch
        for batch_num in batch.unique():
            # Select nodes belonging to the current graph
            nodes = (batch == batch_num).nonzero().squeeze()
            # print("NODES NODES", nodes)
            # print("batch batch", batch)
            # print("shape of edge_index", edge_index.shape)
            for node in nodes:
                # print("Anchor node", node)
                # Extract the k-hop subgraph for the current node

                subset, edge_index_sub, _, _ = k_hop_subgraph(
                    node_idx=node.item(),
                    num_hops=self.k,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=None if batch is None else batch.size(0),
                )
                # Create subgraph batch
                # print("subset", subset)
                # print("edge_index_sub", edge_index_sub)
                subgraph_batch = torch.zeros(
                    subset.size(0), dtype=torch.long, device=edge_index.device
                )

                # Get node features for the subgraph
                x_sub = x_orig[subset]
                # print("originak esm embedding subset of features", x_sub.shape)
                # print("anchor node representation", x[node].shape)
                x_sub = (
                    x[node] * x_sub
                )  # multiplying anchor node representatrion with esm embeddings of rest of the nodes
                # print("subset of features", x_sub)
                # print("edge_index_sub", edge_index_sub)
                # Apply convolution and pooling to the subgraph
                # x_sub = self.conv2(x_sub, edge_index_sub)
                # print("shape of x_sub", x_sub.shape)
                # print("shape of edge_index_sub", edge_index_sub.shape)
                # print("shape of subgraph_batch", subgraph_batch.shape)
                x_sub, edge_index_sub, _, subgraph_batch, perm, score_sub = (
                    self.pool_layer(
                        x=x_sub, edge_index=edge_index_sub, batch=subgraph_batch
                    )
                )
                # print("NOW LOOK AT PERM", perm)
                # print("NOW LOOK AT SUBSET of PERM", subset[perm])
                ##### adding linear layer to score sub
                # print("scoresubshape", score_sub.shape)
                # score_sub = self.linear(score_sub.unsqueeze(1))
                # score_sub = self.sigmoid(score_sub.unsqueeze(1))
                score_sub = self.sigmoid(score_sub)
                # print("SCORE SUB", score_sub)
                # ego_nodes.append(perm) # relabelled nodes
                ego_nodes.append(subset[perm])  # original nodes
                # Store processed subgraphs
                assert len(score_sub) == len(
                    perm
                )  ## node scores length should be equal to number of nodes in subgraph
                assert len(perm) == len(subgraph_batch)
                subgraph_list.append(score_sub)
                subgraph_batch_list.append(subgraph_batch)
        # x_batched, batch_mapping = to_dense_batch(torch.cat(subgraph_list, dim=0), batch=torch.cat(subgraph_batch_list, dim=0))
        return (
            subgraph_list,
            ego_nodes,
        )  # returns list of variable sized tensors with node scores for each ego graph, and nodes as part of ego graph for one batch


class GraphRPN(torch.nn.Module):
    """Graph RPN Model: A GNN model with a pruning unit and a functionality prediction unit"""

    def __init__(self, k, input_dim, hidden_dim, num_classes):
        """_summary_

        Args:
            k (_type_): _description_
            input_dim (_type_): _description_
            hidden_dim (_type_): _description_
            num_classes (_type_): _description_
        """
        super(GraphRPN, self).__init__()
        self.k_layer_gcn = torch.nn.ModuleList(
            [GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(k)]
        )
        self.graph_pruning_unit = PruningUnit(k, input_dim, hidden_dim, num_classes)
        self.functionality_prediction_unit = GATConv(hidden_dim, num_classes)
        self.sigmoid = Sigmoid()
        print("input dim to GRPN", input_dim)
        print("hidden dim to GRPN", hidden_dim)

    def forward(self, x, edge_index, batch):
        """_summary_

        Args:
            data (_type_): _description_


        Returns:
            _type_: _description_
        """
        # x, edge_index = data.x, data.edge_index
        x_orig = x.detach().clone()
        # Apply GCN layers
        for gcn in self.k_layer_gcn:
            # print("inside gcn layer x", x.shape)
            # print("inside gcn layer edge_index", edge_index.shape)
            # print("inside gcn layer batch", batch.shape)
            x = x.to(torch.float32)
            # print("Inside GRPN forward", gcn.lin.weight.dtype, x.dtype)
            x = gcn(
                x=x, edge_index=edge_index
            )  #### TODO: DOES NOT SUPPORT BATCHING NEED TO CHANGE
            # print("printing shape of x after gcn layer", x.shape)
        # Graph pruning and functionality prediction
        node_scores_list, node_list = self.graph_pruning_unit(
            x, x_orig, edge_index, batch
        )
        functionality_logits = self.functionality_prediction_unit(x, edge_index)
        func_probability = self.sigmoid(functionality_logits)
        return node_scores_list, node_list, func_probability
