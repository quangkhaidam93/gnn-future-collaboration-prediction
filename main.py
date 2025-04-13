import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_networkx
from torch_geometric.nn import SAGEConv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from node2vec import Node2Vec
import networkx as nx
import warnings

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


def get_baseline_scores(graph, edges, method):
    """Calculates scores for edges based on a given method."""
    scores = []
    if method == "cn":  # Common Neighbors
        preds = nx.common_neighbor_centrality(graph, ebunch=edges)
        scores = [p for u, v, p in preds]
    elif method == "pagerank":
        pr = nx.pagerank(graph, alpha=0.85)  # Get PageRank scores
        for u, v in edges:
            # Handle potential KeyError if a node in val/test wasn't in training graph
            score = pr.get(u, 0) * pr.get(v, 0)  # Product of scores
            scores.append(score)
    else:
        raise ValueError("Unknown baseline method")
    return np.array(scores)


def evaluate_baseline(data, method):
    """Evaluates a baseline method (CN or PageRank)."""
    # Build graph using ONLY training edges
    # Ensure data.train_pos_edge_index is on CPU and numpy for to_networkx
    train_data = data.__class__()  # Create a temporary data object for conversion
    train_data.edge_index = data.train_pos_edge_index
    train_data.num_nodes = data.num_nodes
    # Need node features or some attribute for to_networkx to work correctly
    # We only care about structure, so add dummy features if needed
    if not hasattr(train_data, "x") or train_data.x is None:
        train_data.x = torch.ones((train_data.num_nodes, 1))  # Dummy features

    # Convert to NetworkX graph (undirected)
    # Note: If your graph is directed, adjust accordingly. Cora is typically treated as undirected.
    try:
        G = to_networkx(train_data, to_undirected=True)
        # Ensure all nodes exist in the graph, even if isolated in training set
        G.add_nodes_from(range(data.num_nodes))
    except Exception as e:
        print(f"Error converting to NetworkX: {e}")
        # Fallback: create graph manually if to_networkx fails
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        train_edges = data.train_pos_edge_index.cpu().numpy().T
        G.add_edges_from(train_edges)

    print(f"Evaluating baseline: {method.upper()}")
    results = {}
    for split in ["val", "test"]:
        pos_edge_index = getattr(data, f"{split}_pos_edge_index").cpu().numpy()
        neg_edge_index = getattr(data, f"{split}_neg_edge_index").cpu().numpy()

        pos_edges = list(zip(pos_edge_index[0], pos_edge_index[1]))
        neg_edges = list(zip(neg_edge_index[0], neg_edge_index[1]))

        # Get scores
        pos_scores = get_baseline_scores(G, pos_edges, method)
        neg_scores = get_baseline_scores(G, neg_edges, method)

        scores = np.concatenate([pos_scores, neg_scores])
        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

        # Normalize scores for AUC/MRR if they vary wildly (optional but often good)
        # Simple min-max scaling
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Calculate metrics
        auc = roc_auc_score(labels, scores)
        precision, recall, f1 = compute_metrics(
            scores, labels
        )  # Use default 0.5 threshold
        mrr = compute_mrr(scores, labels)

        results[split] = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
        }
        print(
            f"... {split.capitalize()} Results - AUC: {auc:.4f}, F1: {f1:.4f}, MRR: {mrr:.4f}"
        )

    return results


def evaluate_node2vec(
    data, dimensions=64, walk_length=30, num_walks=200, workers=4, p=1, q=1
):
    """Trains Node2Vec embeddings and evaluates link prediction using Logistic Regression."""
    print("Evaluating baseline: Node2Vec")
    # Build graph using ONLY training edges (similar to other baselines)
    train_data = data.__class__()
    train_data.edge_index = data.train_pos_edge_index
    train_data.num_nodes = data.num_nodes
    if not hasattr(train_data, "x") or train_data.x is None:
        train_data.x = torch.ones((train_data.num_nodes, 1))  # Dummy

    try:
        # Directed=False for Cora usually
        G_n2v = to_networkx(train_data, to_undirected=True, remove_self_loops=True)
        G_n2v.add_nodes_from(range(data.num_nodes))  # Ensure all nodes are present
    except Exception as e:
        print(f"Error converting to NetworkX for Node2Vec: {e}")
        G_n2v = nx.Graph()
        G_n2v.add_nodes_from(range(data.num_nodes))
        train_edges = data.train_pos_edge_index.cpu().numpy().T
        G_n2v.add_edges_from(train_edges)

    # --- Train Node2Vec ---
    node2vec_model = Node2Vec(
        G_n2v,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
        quiet=True,
    )
    # Train embeddings (use gensim Word2Vec)
    # window=10, min_count=1, sg=1 (skip-gram) are common defaults
    model = node2vec_model.fit(window=10, min_count=1, batch_words=4, sg=1)
    embeddings = model.wv

    # --- Prepare data for Logistic Regression Link Predictor ---
    def get_link_features(edge_index, embeddings, operator="hadamard"):
        features = []
        embs_dim = embeddings.vector_size
        for u, v in edge_index.T:  # Iterate through edges
            # Check if nodes exist in embeddings (trained only on training graph nodes)
            u_str, v_str = str(u), str(v)  # Node2Vec uses string node IDs
            if u_str in embeddings and v_str in embeddings:
                emb_u = embeddings[u_str]
                emb_v = embeddings[v_str]
                if operator == "hadamard":
                    feature = emb_u * emb_v
                elif operator == "l1":
                    feature = np.abs(emb_u - emb_v)
                elif operator == "l2":
                    feature = (emb_u - emb_v) ** 2
                elif operator == "avg":
                    feature = (emb_u + emb_v) / 2.0
                else:  # Default to hadamard
                    feature = emb_u * emb_v
                features.append(feature)
            else:
                # Handle missing nodes: append zero vector or skip
                features.append(np.zeros(embs_dim))
        return np.array(features)

    # Training features for Logistic Regression
    # Use the same negative sampling strategy as GraphSAGE training *if possible*
    # Or, just generate new negatives for simplicity here.
    train_pos_feats = get_link_features(data.train_pos_edge_index.cpu(), embeddings)
    # Generate negative samples for training the classifier
    num_train_pos = data.train_pos_edge_index.size(1)
    train_neg_edge_index = negative_sampling(
        edge_index=data.edge_index,  # Sample from all edges
        num_nodes=data.num_nodes,
        num_neg_samples=num_train_pos,
        method="sparse",
    ).cpu()
    train_neg_feats = get_link_features(train_neg_edge_index, embeddings)

    X_train = np.concatenate([train_pos_feats, train_neg_feats], axis=0)
    y_train = np.concatenate(
        [np.ones(len(train_pos_feats)), np.zeros(len(train_neg_feats))]
    )

    # Train Logistic Regression classifier
    lr = LogisticRegression(solver="liblinear", random_state=42, max_iter=100)
    lr.fit(X_train, y_train)

    # --- Evaluate Node2Vec Link Prediction ---
    results = {}
    for split in ["val", "test"]:
        pos_edge_index = getattr(data, f"{split}_pos_edge_index").cpu()
        neg_edge_index = getattr(data, f"{split}_neg_edge_index").cpu()

        pos_feats = get_link_features(pos_edge_index, embeddings)
        neg_feats = get_link_features(neg_edge_index, embeddings)

        X_eval = np.concatenate([pos_feats, neg_feats], axis=0)
        y_eval = np.concatenate([np.ones(len(pos_feats)), np.zeros(len(neg_feats))])

        # Predict probabilities
        probs = lr.predict_proba(X_eval)[:, 1]  # Probability of class 1 (link exists)

        # Calculate metrics
        auc = roc_auc_score(y_eval, probs)
        precision, recall, f1 = compute_metrics(
            probs, y_eval
        )  # Use default 0.5 threshold
        mrr = compute_mrr(probs, y_eval)

        results[split] = {
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mrr": mrr,
        }
        print(
            f"... {split.capitalize()} Results - AUC: {auc:.4f}, F1: {f1:.4f}, MRR: {mrr:.4f}"
        )

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
