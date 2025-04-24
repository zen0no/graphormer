from copy import deepcopy

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import data
from dgl.nn import pytorch as dgnn


class GraphEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_size, num_layers):
        super().__init__()
        self.atom_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_size

        self.node_proj = nn.Linear(node_dim, hidden_size)
        self.edge_proj = nn.Linear(edge_dim, hidden_size)
        self.encoder = nn.ModuleList([dgnn.EdgeGATConv(hidden_size, hidden_size, hidden_size, num_heads=1, activation=nn.SiLU()) for _ in range(num_layers)])
        self.actiovation = nn.SiLU()

        self.normalizations = nn.ModuleList([nn.BatchNorm1d(hidden_size)])

        self.pooling = dgnn.AvgPooling()

    def forward(self, graph_batch):
        node_feat = graph_batch.ndata["x"]
        nodes = self.node_proj(node_feat)

        edge_feat = graph_batch.edata["x"]
        edges = self.edge_proj(edge_feat)
        for layer in self.encoder:
            message = layer(graph_batch, nodes, edges)
            x += message
            x = F.silu(x)

        return self.pooling(graph_batch, x)


class GraphEncoderWithNodes(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_size, num_layers, num_heads=4):
        super().__init__()
        self.atom_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_size

        self.node_proj = nn.Linear(node_dim, hidden_size)
        self.edge_proj = nn.Linear(edge_dim, hidden_size)
        self.actiovation = nn.SiLU()
        self.encoder = nn.ModuleList(
            [dgnn.EdgeGATConv(hidden_size, hidden_size, hidden_size, num_heads=num_heads, activation=nn.SiLU()) for _ in range(num_layers)]
        )

        self.normalizations = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers)])

        self.pooling = dgnn.AvgPooling()
        self.head = nn.Linear(num_heads * hidden_size, hidden_size)

    def forward(self, graph_batch, nodes_idx):
        node_feat = graph_batch.ndata["x"]
        nodes = self.node_proj(node_feat)

        edge_feat = graph_batch.edata["x"]
        edges = self.edge_proj(edge_feat)
        for layer, bn in zip(self.encoder, self.normalizations):
            x = layer(graph_batch, nodes, edges).permute(0, 2, 1)
            x = bn(x).permute(0, 2, 1)

        x = self.head(x.reshape(x.shape[0], -1))
        # print(x.shape)
        # print(nodes_idx.max())
        return self.pooling(graph_batch, x), x[nodes_idx]


if __name__ == "__main__":
    from dgl import load_graphs

    graphs = load_graphs("/root/data/graphs/task_1/epoch_0/graphs.bin")

    encoder = GraphEncoder(29, 4, 512, 5)

    batch = dgl.batch(graphs[0][:10])

    print(encoder(batch).shape)
