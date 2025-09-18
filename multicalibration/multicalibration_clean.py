import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from wilds import get_dataset
import numpy as np
from collections import Counter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = get_dataset(dataset="ogb-molpcba", download=True)
data_train = dataset.get_subset('train', transform=None)
data_val = dataset.get_subset('val', transform=None)
data_test = dataset.get_subset('test', transform=None)


# Y is [N, 128], with -1 for missing labels
y_all = dataset.y_array[data_train.indices]
label_counts = (y_all != -1).sum(axis=0)
target_idx = int(label_counts.argmax())
print(f"Chosen target index: {target_idx}, with {label_counts[target_idx]} labels")


def get_scaffolds(dataset):
    scaffolds = []
    # Scaffold is at index 0 in metadata_array
    
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # Handle ConcatDataset by iterating through each dataset
        for i in range(len(dataset)):
            # Find which dataset this index belongs to
            cumulative_length = 0
            for sub_dataset in dataset.datasets:
                if i < cumulative_length + len(sub_dataset):
                    # This index belongs to sub_dataset
                    local_idx = i - cumulative_length
                    if hasattr(sub_dataset, 'dataset'):  # WILDSSubset
                        scaffold_label = sub_dataset.dataset.metadata_array[sub_dataset.indices[local_idx], 0].item()
                    else:  # Direct dataset
                        scaffold_label = sub_dataset.dataset.metadata_array[local_idx, 0].item()
                    scaffolds.append(scaffold_label)
                    break
                cumulative_length += len(sub_dataset)
    else:
        # Handle regular WILDSSubset
        scaffold_idx = 0
        for i in range(len(dataset)):
            scaffold_label = dataset.dataset.metadata_array[dataset.indices[i], scaffold_idx].item()
            scaffolds.append(scaffold_label)
    
    return scaffolds

# Combine all datasets
all_data = torch.utils.data.ConcatDataset([data_train, data_val, data_test])

# Get scaffolds for all data
all_scaffolds = get_scaffolds(all_data)
scaff_counts = Counter(all_scaffolds)
top5_scaffs = {s for s, _ in scaff_counts.most_common(5)}

print(f"Top 5 scaffolds: {top5_scaffs}")
print(f"Scaffold counts: {dict(scaff_counts.most_common(5))}")

# Filter all data by top 5 scaffolds
def filter_by_scaffold(dataset, scaffolds):
    idxs = [i for i, s in enumerate(get_scaffolds(dataset)) if s in scaffolds]
    return torch.utils.data.Subset(dataset, idxs)

all_filt = filter_by_scaffold(all_data, top5_scaffs)
print(f"Original total size: {len(all_data)}, Filtered total size: {len(all_filt)}")

# Get scaffold labels for the filtered indices
all_scaffolds = get_scaffolds(all_data)
all_filt_scaffolds = [all_scaffolds[i] for i in range(len(all_data)) if all_scaffolds[i] in top5_scaffs]

# Make a custom dataset that includes scaffold information
class ScaffoldDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scaffold_labels):
        self.dataset = dataset
        self.scaffold_labels = scaffold_labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) == 2:
            data, metadata = item
            return data, metadata, self.scaffold_labels[idx]
        else:
            return item, None, self.scaffold_labels[idx]

scaffold_dataset = ScaffoldDataset(all_filt, all_filt_scaffolds)


total_filt = len(scaffold_dataset)
train_size = int(0.8 * total_filt)
val_size = int(0.1 * total_filt)
test_size = total_filt - train_size - val_size

indices = list(range(total_filt))
import random
random.seed(42)
random.shuffle(indices)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

train_filt = torch.utils.data.Subset(scaffold_dataset, train_indices)
val_filt = torch.utils.data.Subset(scaffold_dataset, val_indices)
test_filt = torch.utils.data.Subset(scaffold_dataset, test_indices)

# print(f"New split sizes - Train: {len(train_filt)}, Val: {len(val_filt)}, Test: {len(test_filt)}")


class GIN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINConv(nn2)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # Ensure correct types
        x = x.float()
        edge_index = edge_index.long()
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = global_add_pool(h, batch)
        return torch.sigmoid(self.lin(h)).view(-1)

# Get input dim
sample = train_filt[0]
print(f"Sample type: {type(sample)}")
print(f"Sample length: {len(sample) if isinstance(sample, tuple) else 'not tuple'}")
if isinstance(sample, tuple):
    print(f"Sample contents: {[type(x) for x in sample]}")

if isinstance(sample, tuple) and len(sample) == 3:
    # ScaffoldDataset returns (data, metadata, scaffold)
    sample_data = sample[0]
elif isinstance(sample, tuple) and len(sample) == 2:
    # WILDS returns (data, metadata) tuple
    sample_data = sample[0]
else:
    sample_data = sample

# Handle the case where sample_data is still a tuple (data, metadata)
if isinstance(sample_data, tuple):
    sample_data = sample_data[0]  # Extract just the data

print(f"Sample data type: {type(sample_data)}")
in_dim = sample_data.x.shape[1]
model = GIN(in_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def collate(batch):
    from torch_geometric.data import Batch
    # Handle ScaffoldDataset which returns (data, metadata, scaffold) tuples
    data_list = []
    scaffold_list = []
    
    for item in batch:
        if isinstance(item, tuple) and len(item) == 3:
            # ScaffoldDataset format: (data, metadata, scaffold)
            data, metadata, scaffold = item
            # Extract the actual Data object from the data tuple
            if isinstance(data, tuple):
                data = data[0]  # Extract Data from (data, metadata)
            data_list.append(data)
            scaffold_list.append(scaffold)
        elif isinstance(item, tuple) and len(item) == 2:
            # Original WILDS format: (data, metadata)
            data, metadata = item
            data_list.append(data)
            scaffold_list.append(0)  # Default scaffold
        else:
            # Direct data format
            data_list.append(item)
            scaffold_list.append(0)  # Default scaffold
    
    batch_data = Batch.from_data_list(data_list)
    batch_data.scaffold_labels = torch.tensor(scaffold_list)
    return batch_data

train_loader = torch.utils.data.DataLoader(train_filt, batch_size=64, shuffle=True, collate_fn=collate)
val_loader = torch.utils.data.DataLoader(val_filt, batch_size=64, shuffle=False, collate_fn=collate)
test_loader = torch.utils.data.DataLoader(test_filt, batch_size=64, shuffle=False, collate_fn=collate)

bce = nn.BCELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Get y values from the batch
        y = batch.y[:, target_idx].float().to(device)
        mask = (y != -1)
        if mask.sum() == 0:
            continue
        opt.zero_grad()
        pred = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device))
        # Convert to binary: 0 for negative, 1 for positive
        y_binary = (y[mask] > 0).float()
        loss = bce(pred[mask], y_binary)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, loss {total_loss/len(train_loader):.4f}")

def get_preds(loader):
    model.eval()
    all_p, all_y, scaffs = [], [], []
    total_samples = 0
    valid_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            y = batch.y[:, target_idx].float()
            mask = (y != -1)
            total_samples += len(y)
            valid_samples += mask.sum().item()
            
            if mask.sum() == 0:
                continue
                
            pred = model(batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device)).cpu()
            # Convert to binary: 0 for negative, 1 for positive
            y_binary = (y[mask] > 0).float()
            all_p.append(pred[mask])
            all_y.append(y_binary)
            
            # Get scaffold labels from the batch
            if hasattr(batch, 'scaffold_labels'):
                batch_scaffolds = batch.scaffold_labels.cpu().numpy()
            else:
                # Fallback: use zeros
                batch_scaffolds = np.zeros(len(batch.x))
            scaffs.extend(batch_scaffolds)
    
    print(f"Total samples processed: {total_samples}, Valid samples: {valid_samples}")
    
    if len(all_p) == 0:
        print("Warning: No valid predictions found!")
        return np.array([]), np.array([]), np.array([])
    
    return torch.cat(all_p).numpy(), torch.cat(all_y).numpy(), np.array(scaffs)

p_val, y_val, scaff_val = get_preds(val_loader)
p_test, y_test, scaff_test = get_preds(test_loader)

def multicalibrate(p, y, groups, eps=0.01, n_min=20, eta=1.0):
    p_adj = p.copy()
    unique_groups = np.unique(groups)
    bins = np.linspace(0, 1, 11)
    for _ in range(10):  # epochs
        max_violation = 0
        for g in unique_groups:
            mask_g = (groups == g)
            for i in range(len(bins)-1):
                lo, hi = bins[i], bins[i+1]
                mask_b = (p_adj >= lo) & (p_adj < hi)
                idx = np.where(mask_g & mask_b)[0]
                if len(idx) < n_min:
                    continue
                y_bar = y[idx].mean()
                p_bar = p_adj[idx].mean()
                delta = y_bar - p_bar
                if abs(delta) > eps:
                    p_adj[idx] = np.clip(p_adj[idx] + eta * delta, 0, 1)
                    max_violation = max(max_violation, abs(delta))
        if max_violation <= eps:
            break
    return p_adj

p_val_mc = multicalibrate(p_val, y_val, scaff_val)
# Learn mapping raw->adj for deployment
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
Xg_val = enc.fit_transform(scaff_val.reshape(-1,1))
X_val = np.hstack([p_val.reshape(-1,1), Xg_val])
reg = LinearRegression().fit(X_val, p_val_mc)


def apply_mc(p_raw, groups):
    Xg = enc.transform(groups.reshape(-1,1))
    X = np.hstack([p_raw.reshape(-1,1), Xg])
    return np.clip(reg.predict(X), 0, 1)


def ece_score(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0: 
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum()/len(y_true)) * abs(acc - conf)
    return ece

ece_before = ece_score(y_test, p_test)
p_test_mc = apply_mc(p_test, scaff_test)
ece_after = ece_score(y_test, p_test_mc)

print(f"ECE before MC: {ece_before:.4f}, after MC: {ece_after:.4f}")
