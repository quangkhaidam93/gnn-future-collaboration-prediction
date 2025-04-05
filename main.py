import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges, negative_sampling
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Load and preprocess dataset
dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())
data = dataset[0]
data = train_test_split_edges(data)


# GraphSAGE model class
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# Link predictor class
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(in_channels * 2, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, 1),
        )

    def forward(self, x_i, x_j):
        z = torch.cat([x_i, x_j], dim=1)
        return torch.sigmoid(self.lin(z))


# Label creation for edges
def get_link_labels(pos_edge_index, neg_edge_index):
    num_pos = pos_edge_index.size(1)
    num_neg = neg_edge_index.size(1)
    labels = torch.zeros(num_pos + num_neg)
    labels[:num_pos] = 1.0
    return labels


# Training loop
def train(model, predictor, data, optimizer):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.train_pos_edge_index)
    pos_edge = data.train_pos_edge_index
    neg_edge = negative_sampling(
        edge_index=pos_edge,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge.size(1),
        method="sparse",  # or 'dense' if your graph is small
    )

    pos_pred = predictor(z[pos_edge[0]], z[pos_edge[1]])
    neg_pred = predictor(z[neg_edge[0]], z[neg_edge[1]])

    preds = torch.cat([pos_pred, neg_pred], dim=0).squeeze()
    labels = get_link_labels(pos_edge, neg_edge)

    loss = F.binary_cross_entropy(preds, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


def compute_metrics(preds, labels, threshold=0.5):
    """Compute Precision, Recall, F1-Score with threshold"""
    preds = (preds > threshold).cpu().numpy()  # Apply threshold to convert to binary
    labels = labels.cpu().numpy()

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    return precision, recall, f1


def compute_mrr(preds, labels):
    """Compute Mean Reciprocal Rank (MRR)"""
    ranks = []
    for i, pred in enumerate(preds):
        # If the prediction is a true positive, its rank is i+1 (1-indexed)
        if labels[i] == 1:
            ranks.append(1 / (i + 1))  # 1-indexed rank
    if len(ranks) > 0:
        mrr = np.mean(ranks)
    else:
        mrr = 0.0
    return mrr


# Evaluation function
@torch.no_grad()
def test(model, predictor, data):
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)

    results = {}
    for split in ["val", "test"]:
        pos_edge = getattr(data, f"{split}_pos_edge_index")
        neg_edge = getattr(data, f"{split}_neg_edge_index")

        pos_pred = predictor(z[pos_edge[0]], z[pos_edge[1]])
        neg_pred = predictor(z[neg_edge[0]], z[neg_edge[1]])

        preds = torch.cat([pos_pred, neg_pred], dim=0).squeeze()
        labels = get_link_labels(pos_edge, neg_edge)

        # Calculate AUC
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(labels.cpu(), preds.cpu())

        # Calculate Precision, Recall, F1-Score
        precision, recall, f1 = compute_metrics(preds, labels)

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = compute_mrr(preds, labels)

        # Store results
        results[split] = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
        }

    return results


# Initialize and train the model
model = GraphSAGE(in_channels=dataset.num_features, hidden_channels=64)
predictor = LinkPredictor(in_channels=64)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(predictor.parameters()), lr=0.01
)

for epoch in range(1, 101):
    loss = train(model, predictor, data, optimizer)
    if epoch % 10 == 0:
        results = test(model, predictor, data)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
            f'Val AUC: {results["val"]["auc"]:.4f}, Val Precision: {results["val"]["precision"]:.4f}, '
            f'Val Recall: {results["val"]["recall"]:.4f}, Val F1: {results["val"]["f1"]:.4f}, '
            f'Val MRR: {results["val"]["mrr"]:.4f}, '
            f'Test AUC: {results["test"]["auc"]:.4f}, Test Precision: {results["test"]["precision"]:.4f}, '
            f'Test Recall: {results["test"]["recall"]:.4f}, Test F1: {results["test"]["f1"]:.4f}, '
            f'Test MRR: {results["test"]["mrr"]:.4f}'
        )
