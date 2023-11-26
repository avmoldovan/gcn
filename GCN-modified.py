import numpy as np
# Import necessary libraries
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx

# Load the Karate Club Dataset
dataset = KarateClub()
data = dataset[0]

# Convert to NetworkX graph for visualization
G = to_networkx(data, to_undirected=True)

# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Prepare for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# Function to add Gaussian noise to node features
def add_noise_to_features(features, noise_level=0.1):
    noise = torch.randn(features.size()) * noise_level
    return features + noise.to(device)

# Function to randomly perturb edges
def perturb_edges(edge_index, perturb_rate=0.1):
    edges = edge_index.t()
    num_perturb = int(edges.shape[0] * perturb_rate) #int(edges.shape[0] * perturb_rate)
    perturbed_edges = edges.clone()

    for _ in range(num_perturb):
        idx = np.random.randint(edges.shape[0])
        perturbed_edges[idx] = torch.tensor(np.random.choice(edges.shape[1], 2), dtype=torch.long).to(device)

    return torch.tensor(perturbed_edges, dtype=torch.long).t()

# Create a noisy copy of the data
noisy_data = data.clone()
noisy_data.x = add_noise_to_features(noisy_data.x)
noisy_data.edge_index = perturb_edges(noisy_data.edge_index)


# Train a model
def train_model(data, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    return model

# Train models on original and noisy data
original_model = GCN().to(device)
noisy_model = GCN().to(device)
noisy_data = noisy_data.to(device)

train_model(data, original_model)
train_model(noisy_data, noisy_model)


def visualize_graph(G, model, data, title):
    model.eval()
    _, pred = model(data).max(dim=1)
    color = pred.cpu().numpy()

    plt.figure(figsize=(8, 8))
    nx.draw_networkx(G, node_color=color, with_labels=True, node_size=70, font_size=10)
    plt.title(title)

# Visualize original and noisy graphs
visualize_graph(G, original_model, data, "Original Network")
visualize_graph(G, noisy_model, noisy_data, "Noisy Network")
plt.show()