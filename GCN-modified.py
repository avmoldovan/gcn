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

# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
# node_features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # Example node features


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

print('Dataset properties')
print('==============================================================')
print(f'Dataset: {dataset}') #This prints the name of the dataset
print(f'Number of graphs in the dataset: {len(dataset)}')
print(f'Number of features: {dataset.num_features}') #Number of features each node in the dataset has
print(f'Number of classes: {dataset.num_classes}') #Number of classes that a node can be classified into


#Since we have one graph in the dataset, we will select the graph and explore it's properties

dsmeta = dataset[0]
print('Graph properties')
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {dsmeta.num_nodes}') #Number of nodes in the graph
print(f'Number of edges: {dsmeta.num_edges}') #Number of edges in the graph
print(f'Average node degree: {dsmeta.num_edges / dsmeta.num_nodes:.2f}') # Average number of nodes in the graph
print(f'Contains isolated nodes: {dsmeta.has_isolated_nodes()}') #Does the graph contains nodes that are not connected
print(f'Contains self-loops: {dsmeta.has_self_loops()}') #Does the graph contains nodes that are linked to themselves
print(f'Is undirected: {dsmeta.is_undirected()}') #Is the graph an undirected graph

# Function to add Gaussian noise to node features
def add_noise_to_features(features, noise_level=0.1):
    #torch.special.entr(features)
    noise = torch.randn(features.size()) * noise_level
    return features + noise.to(device)

# Initialize dictionaries to store history
history = {
    'original': {'features': [], 'edges': []},
    'noisy': {'features': [], 'edges': []}
}

# Function to randomly perturb edges
def perturb_edges(edge_index, perturb_rate=0.1):
    edges = edge_index.t()
    num_perturb = int(edges.shape[0] * perturb_rate) #int(edges.shape[0] * perturb_rate)
    perturbed_edges = edges.clone()

    for _ in range(num_perturb):
        idx = np.random.randint(edges.shape[0])
        #entr = torch.special.entr(edges).to(device)
        #torch.special.entr(edges).to(device)  #
        perturbed_edges[idx] = torch.tensor(np.random.choice(edges.shape[1], 2), dtype=torch.long).to(device)

    return torch.tensor(perturbed_edges, dtype=torch.long).t()

def update_features_with_edge_values(data, G):
    edge_index = data.edge_index
    features = data.x.clone()

    # Iterate over each edge
    for i in range(edge_index.shape[1]):
        node1 = edge_index[0, i].item()
        node2 = edge_index[1, i].item()

        # Sum the features of the two nodes
        feature_sum = features[node1] + features[node2]
        entr = torch.special.entr(features[node1] + features[node2])

        G[node1][node2]['weight'] = entr
        # Update the features of both nodes
        # features[node1] = feature_sum
        # features[node2] = feature_sum

    return features

# Create a noisy copy of the data
noisy_data = data.clone()
#previous approach with two different sets of operations
#noisy_data.x = add_noise_to_features(noisy_data.x)
#noisy_data.edge_index = perturb_edges(noisy_data.edge_index)

#data.x          = update_features_with_edge_values(data)
noisy_data.x    = update_features_with_edge_values(noisy_data, G)

# Train a model
# def train_model(data, model, epochs = 200):
#     best_loss = 0.
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         out = model(data)
#         loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#         #if epoch % 10 == 0:
#         #    print(f'Epoch: {epoch}, Loss: {loss}')
#
#         if best_loss < loss:
#             best_loss = loss
#         elif best_loss == 0.:
#             best_loss = loss
#
#     print(f'Epoch: {epoch}, Loss: {best_loss}')
#     return model, best_loss

def train_model_with_history(data, model, history_dict, history_key, epochs = 200):
    best_loss = 0.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Record the feature and edge values
        history_dict[history_key]['features'].append(data.x.cpu().detach().numpy())
        history_dict[history_key]['edges'].append(data.edge_index.cpu().detach().numpy())

        if best_loss < loss:
            best_loss = loss
        elif best_loss == 0.:
            best_loss = loss

    print(f'Epoch: {epoch}, Loss: {best_loss}')
    return model, best_loss


# Train models on original and noisy data
original_model = GCN().to(device)
noisy_model = GCN().to(device)
noisy_data = noisy_data.to(device)

runs = 30
epcount = 200
repochs = range(0, epcount)
unmodified_wins = 0

# Train models on original and noisy data
for j in range(runs):
    print("Training unmodified model")
    _, modelloss1 = train_model_with_history(data, original_model, history, 'original', epcount)
    #_, modelloss1 = train_model(data, original_model, epcount)
    print("Training MODIFIED model")
    #_, modelloss2 = train_model(noisy_data, noisy_model, epcount)
    _, modelloss2 = train_model_with_history(noisy_data, noisy_model, history, 'noisy', epcount)

    if (modelloss1 < modelloss2):
        print(f'UNMODIFIED model is better')
        unmodified_wins+=1
    else:
        print(f'ALTERED model is better')

def calculate_averages(history):
    feature_averages = [np.mean(features, axis=0) for features in history['features']]
    edge_counts = [edges.shape[1] for edges in history['edges']]
    return feature_averages, edge_counts

original_feature_averages, original_edge_counts = calculate_averages(history['original'])
noisy_feature_averages, noisy_edge_counts = calculate_averages(history['noisy'])

def plots():
    plt.figure(figsize=(12, 5))
    # Plotting average feature values
    plt.subplot(1, 2, 1)
    plt.plot(repochs, [np.mean(f) for f in original_feature_averages], label='Original')
    plt.plot(repochs, [np.mean(f) for f in noisy_feature_averages], label='Noisy')
    plt.title('Average Feature Values Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Average Feature Value')
    plt.legend()

    # Plotting edge counts
    plt.subplot(1, 2, 2)
    plt.plot(repochs, original_edge_counts, label='Original')
    plt.plot(repochs, noisy_edge_counts, label='Noisy')
    plt.title('Edge Counts Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Edges')
    plt.legend()

    plt.tight_layout()
    plt.show()

#plots()

# accessing historical data
first_epoch_features_original = history['original']['features'][0]
first_epoch_edges_original = history['original']['edges'][0]


print(f'ALTERED model won {int(runs - unmodified_wins)} times out of {runs}')

def visualize_graph(G, model1, model2, data1, data2, title1, title2):
    fig, ax = plt.subplots(2, figsize=(12, 12))

    model1.eval()
    _, pred = model1(data1).max(dim=1)
    color = pred.cpu().numpy()

    ax[0].set_title(title1)
    nx.draw_networkx(G, node_color=color, with_labels=True, node_size=70, font_size=10, ax=ax[0])

    model2.eval()
    _, pred = model2(data2).max(dim=1)
    color = pred.cpu().numpy()

    ax[1].set_title(title2)
    nx.draw_networkx(G, node_color=color, with_labels=True, node_size=70, font_size=10, ax=ax[1])

    plt.show()

# Visualize original and noisy graphs
#visualize_graph(G, original_model, noisy_model, data, noisy_data, "Original Network", "Noisy Network")
