import sys

sys.path.insert(0, "../HEAL")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from HEAL.pool import GraphMultisetTransformer

# from pool_pen import GraphMultisetTransformer as GraphMultisetTransformerPen
# from vcr_graphformer import TransformerModel
from torch_geometric.nn import global_max_pool as gmp


"Modified from HEAL: https://github.com/ZhonghuiGu/HEAL/tree/main"


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.mlp(x)


class GraphCNN(nn.Module):
    def __init__(
        self,
        channel_dims=[512, 512, 512],
        fc_dim=512,
        num_classes=256,
        pooling="MTP",
        polynormer=False,
    ):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [
            GCNConv(gcn_dims[i - 1], gcn_dims[i], bias=True)
            for i in range(1, len(gcn_dims))
        ]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            if not polynormer:
                self.pool = GraphMultisetTransformer(
                    512,
                    256,
                    512,
                    None,
                    10000,
                    0.25,
                    ["GMPool_G", "GMPool_G"],
                    num_heads=8,
                    layer_norm=True,
                )
            else:
                # self.pool = Polynormer(
                #     512,
                #     256,
                #     512,
                #     local_layers=3,
                #     global_layers=2,
                #     in_dropout=0.15,
                #     dropout=0.5,
                #     global_dropout=0.5,
                #     heads=8,
                #     beta=-1,
                #     pre_ln=False,
                # )
                pass
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch, pertubed=False):
        # x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, edge_index.long()))
            else:
                x = x + F.relu(gcn_layer(x, edge_index.long()))

            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        if self.pooling == "MTP":
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, batch, edge_index.long())
        else:
            g_level_feat = self.pool(x, batch)

        n_level_feat = x

        return n_level_feat, g_level_feat


class CL_protNET(torch.nn.Module):
    def __init__(
        self, out_dim, esm_embed=True, pooling="MTP", pertub=False, polynormer=False
    ):
        super(CL_protNET, self).__init__()
        self.esm_embed = esm_embed
        self.pertub = pertub
        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512)
        self.pooling = pooling
        # self.proj_spot = nn.Linear(19, 512)
        if esm_embed:
            self.proj_esm = nn.Linear(1280, 512)
            self.gcn = GraphCNN(pooling=pooling, polynormer=polynormer)
        else:
            self.gcn = GraphCNN(pooling=pooling, polynormer=polynormer)
        # self.esm_g_proj = nn.Linear(1280, 512)
        self.readout = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, out_dim),
            nn.Sigmoid(),
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, native_x, edge_index, batch):

        x_aa = self.one_hot_embed(native_x.long())
        x_aa = self.proj_aa(x_aa)

        if self.esm_embed:
            x = x.float()
            x_esm = self.proj_esm(x)  # [B, 1280] -> [B, 512]
            # print("x_esm", x_esm.shape)
            # print("x_aa", x_aa.shape)
            x = F.relu(x_aa + x_esm)

        else:
            x = F.relu(x_aa)

        gcn_n_feat1, gcn_g_feat1 = self.gcn(
            x, edge_index, batch
        )  # graph cnn input [B, 512]
        # print("gcn_g_feat1", gcn_g_feat1.shape)
        # print("self readout shaoe", self.readout)
        # gcn_g_feat1 = self.mamba(gcn_g_feat1, edge_index, edge_attr=None, batch=batch)
        if self.pertub:
            gcn_n_feat2, gcn_g_feat2 = self.gcn(x, edge_index, batch, pertubed=True)

            y_pred = self.readout(gcn_g_feat1)

            return y_pred, gcn_g_feat1, gcn_g_feat2
        else:
            y_pred = self.readout(gcn_g_feat1)

            return y_pred
