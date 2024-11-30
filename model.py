import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import reuters
from nltk import download
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomAttention(nn.Module):
    def __init__(self, d_model, attention_type='multihead', alignment_fn='scaled_dot', n_heads=1, window_size=5, temperature=1.0):
        super(CustomAttention, self).__init__()
        self.d_model = d_model
        self.attention_type = attention_type
        self.alignment_fn = alignment_fn
        self.n_heads = n_heads
        self.window_size = window_size
        self.temperature = nn.Parameter(torch.tensor(temperature))  # For adaptive attention

        # Linear layers for query, key, value (used in most attention mechanisms)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        if attention_type == 'multihead':
            self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def compute_alignment(self, query, key):
        if self.alignment_fn == 'dot':
            return torch.matmul(query, key.transpose(-2, -1))
        elif self.alignment_fn == 'scaled_dot':
            return torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
        elif self.alignment_fn == 'additive':
            score = torch.tanh(query.unsqueeze(-2) + key.unsqueeze(-3))
            return score.sum(dim=-1)
        else:
            raise ValueError(f"Unknown alignment function: {self.alignment_fn}")

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        if self.attention_type == 'multihead':
            return self.attention(query, key, value)

        elif self.attention_type == 'local':
            batch_size, seq_len, _ = x.size()
            outputs = []
            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                local_query = query[:, i:i+1, :]
                local_key = key[:, start:end, :]
                local_value = value[:, start:end, :]
                alignment = self.compute_alignment(local_query, local_key)
                weights = F.softmax(alignment, dim=-1)
                outputs.append(torch.matmul(weights, local_value).squeeze(1))
            return torch.stack(outputs, dim=1) , weights

        elif self.attention_type == 'global':
            alignment = self.compute_alignment(query, key)
            weights = F.softmax(alignment, dim=-1)
            return torch.matmul(weights, value),  weights

        elif self.attention_type == 'adaptive':
            # Adaptive attention using learnable temperature
            alignment = self.compute_alignment(query, key)
            alignment /= self.temperature  # Apply the temperature to control sharpness of softmax
            weights = F.softmax(alignment, dim=-1)
            return torch.matmul(weights, value),weights

        elif self.attention_type == 'stochastic':
            # Stochastic attention: Sample based on attention weights
            alignment = self.compute_alignment(query, key)
            weights = F.softmax(alignment, dim=-1)  # Shape: (batch_size, seq_len, seq_len)

            # Ensure that we are sampling from the correct dimension (along the last dimension for each batch)
            batch_size, seq_len, _ = weights.size()

            # Reshape weights to 2D so we can use torch.multinomial for each batch
            sampled_indices = torch.multinomial(weights.view(batch_size * seq_len, -1), num_samples=1)

            # Reshape indices back to (batch_size, seq_len, 1)
            sampled_indices = sampled_indices.view(batch_size, seq_len, 1)

            # Gather the corresponding values for the sampled indices
            sampled_values = torch.gather(value, dim=1, index=sampled_indices.expand(-1, -1, value.size(-1)))

            return sampled_values,weights

        elif self.attention_type == 'kernelized':
            query = F.elu(query) + 1
            key = F.elu(key) + 1
            alignment = torch.matmul(query, key.transpose(-2, -1))
            weights = F.softmax(alignment, dim=-1)
            return torch.matmul(weights, value), weights

        elif self.attention_type == 'group_query':
            # Group query logic
            groups = query.chunk(self.n_heads, dim=1)
            grouped_results = []
            for g in groups:
                alignment = self.compute_alignment(g, key)
                weights = F.softmax(alignment, dim=-1)
                grouped_results.append(torch.matmul(weights, value))
            return torch.cat(grouped_results, dim=1), weights

        elif self.attention_type == 'hierarchical':
            # Hierarchical attention
            chunks = x.chunk(4, dim=1)  # Dividing into hierarchical levels
            hierarchical_outputs = []
            for chunk in chunks:
                query = self.query(chunk)
                key = self.key(chunk)
                value = self.value(chunk)
                alignment = self.compute_alignment(query, key)
                weights = F.softmax(alignment, dim=-1)
                hierarchical_outputs.append(torch.matmul(weights, value))
            return torch.cat(hierarchical_outputs, dim=1), weights

        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_hidden_dim, dropout, attention_type, alignment_fn):
        super(TransformerBlock, self).__init__()
        self.attention = CustomAttention(d_model, attention_type=attention_type, alignment_fn=alignment_fn, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output , _ = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DocumentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, ff_hidden_dim, num_layers, num_classes, max_seq_len, dropout=0.1, attention_type='multihead', alignment_fn='scaled_dot'):
        super(DocumentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_hidden_dim, dropout, attention_type, alignment_fn)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)





# Download Reuters dataset
download('reuters')
download('punkt')

# Load Reuters Dataset
docs = reuters.fileids()
documents = [reuters.raw(doc_id) for doc_id in docs]
labels = [reuters.categories(doc_id) for doc_id in docs]

# Binarize the labels for multi-label classification
mlb = MultiLabelBinarizer()
labels_binarized = mlb.fit_transform(labels)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels_binarized, test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Custom Dataset
class ReutersDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        features = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': features['input_ids'].squeeze(0),
            'attention_mask': features['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }



# Training Loop
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation Loop
def evaluate(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids)

            # Calculate loss
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            # Get predictions (threshold at 0.5 for multi-label classification)
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_valid_loss = valid_loss / len(valid_loader)

    # Concatenate predictions and labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return avg_valid_loss, metrics



vocab_size = tokenizer.vocab_size+1
d_model = 128
n_heads = 8
ff_hidden_dim = 512
num_layers = 4
num_classes =len(mlb.classes_)
max_seq_len = 2048
batch_size = 64
device = 'cuda'
window_size = 5
criterion = nn.BCEWithLogitsLoss()


train_dataset = ReutersDataset(X_train, y_train, tokenizer)
test_dataset = ReutersDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size, shuffle=False)


import argparse

parser = argparse.ArgumentParser(description="A simple script to demonstrate command-line arguments in Python.")
parser.add_argument('--attn_type', type=str, help="Your name")  # Positional argument
parser.add_argument('--align_fn', type=str, help="Your age", required=False)  # Optional argument
parser.add_argument("--cuda" ,type=str)
args = parser.parse_args()

attn_type = args.attn_type
align_fn = args.align_fn
device = args.cuda
epochs = 100

print(f"Testing with Attention: {attn_type}, Alignment: {align_fn}")
model = DocumentClassifier(
    vocab_size,
    d_model,
    n_heads,
    ff_hidden_dim,
    num_layers,
    num_classes,
    max_seq_len,
    attention_type=attn_type,
    alignment_fn=align_fn
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
accuracy = 0
import matplotlib.pyplot as plt
import os

# Create directory for saving graphs
save_dir = f"models/{attn_type}_{align_fn}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

chk = torch.load(f"{save_dir}/best_model_{attn_type}_{align_fn}.pth" ,map_location= device )
model.load_state_dict(chk)

text = documents[0][10:20]

text_tokens = tokenizer.tokenize(text)
x = tokenizer(text, return_tensors='pt')['input_ids'].to(device)
seq_len = x.size(1)
x = model.embedding(x) + model.pos_embedding[:, :seq_len, :]
_ , weights = model.layers[0].attention(x)
print(text_tokens)


import matplotlib.pyplot as plt
import seaborn as sns
def plot_attention_heatmap(attention_weights, input_tokens=None, title="Attention Heatmap"):
    """
    Plots a heatmap for the given attention weights.

    Args:
        attention_weights (torch.Tensor): Attention weights of shape (batch_size, seq_len, seq_len).
        input_tokens (list[str], optional): List of input tokens corresponding to the sequence length. 
                                            If None, will use generic indices.
        title (str): Title for the heatmap.
    """
    # Convert tensor to numpy array
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.squeeze(0).detach().cpu().numpy()  # Use batch[0]

    # Set up input tokens or default indices
    if input_tokens is None:
        seq_len = attention_weights.shape[0]
        input_tokens = [f"Token {i}" for i in range(seq_len)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, xticklabels=input_tokens, yticklabels=input_tokens, 
                cmap='viridis', cbar=True, square=True, annot=False)
    plt.title(title, fontsize=16)
    plt.xlabel("Keys", fontsize=12)
    plt.ylabel("Queries", fontsize=12)
    plt.savefig(save_dir + "/attention_heatmap.png")
print(weights.shape)
if(weights.shape == 4):
    weights = weights[0]

np.save('weifhts/'+ f"{attn_type}_{align_fn}.npy" , weights.detach().cpu().numpy())
plot_attention_heatmap(weights[0], text_tokens, title=f"{attn_type} {align_fn} Attention Heatmap")







# Initialize lists to store the metrics for each epoch
# def logger(path , data):
#     try:
#         with open(path, 'a') as f:
#          f.write(data + "\n")
#     except:
#         with open(path, 'w') as f:
#          f.write(data + "\n")
# train_losses = []
# valid_losses = []
# accuracies = []
# precisions = []
# recalls = []
# f1_scores = []

# best_accuracy = 0.0  # To track the best accuracy for saving the model

# # Training loop
# from datetime import datetime
# for epoch in range(epochs):
#     # Train the model
#    # torch.cuda.reset_max_memory_allocated()
#     start_time = datetime.now()
#     train_loss = train(model, train_loader, criterion, optimizer, device)
#     data = f"Epoch {epoch+1}, Training Loss: {train_loss:.4f}, "
#     train_losses.append(train_loss)
    
#     # Evaluate on validation data
#     avg_valid_loss, metrics = evaluate(model, valid_loader, criterion, device)
#     valid_losses.append(avg_valid_loss)
    
#     # Store validation metrics
#     accuracies.append(metrics['accuracy'])
#     precisions.append(metrics['precision'])
#     recalls.append(metrics['recall'])
#     f1_scores.append(metrics['f1_score'])
#     end_time = datetime.now() - start_time
#     end_time = end_time.total_seconds()
#     data += f"Validation Loss: {avg_valid_loss:.4f},"
#     data += f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1_score']:.4f}, "
#     data += f"Time taken: {end_time} sec, "
#     data += (f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2} MB")
#     logger(f"{save_dir}/metrics.txt" , data)
#     # Save the model if it improves
#     if best_accuracy < metrics['accuracy']:
#         best_accuracy = metrics['accuracy']
#         torch.save(model.state_dict(), f"{save_dir}/best_model_{attn_type}_{align_fn}.pth")
#         print("Model saved!")

# # Save the plots after training

# # Plot and save training and validation loss
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt

# # Combine all the metrics into one plot
# plt.figure(figsize=(12, 8))

# # Plot Training and Validation Loss
# plt.plot(train_losses, label="Training Loss")
# plt.plot(valid_losses, label="Validation Loss", linestyle='--')

# # Plot Validation Accuracy, Precision, Recall, and F1 Score
# plt.plot(accuracies, label="Validation Accuracy", linestyle='-.')
# plt.plot(precisions, label="Validation Precision", linestyle=':')
# plt.plot(recalls, label="Validation Recall", linestyle='-')
# plt.plot(f1_scores, label="Validation F1 Score", linestyle='--')

# # Set x-axis to show integer epochs only
# plt.xticks(range(0, epochs))

# # Add labels and title
# plt.xlabel("Epochs")
# plt.ylabel("Metrics")
# plt.title(f"Training and Validation Metrics Over Epochs for Attention {attn_type} Alignment Function{align_fn}")
# plt.legend()

# # Enable grid for better readability
# plt.grid(True)

# # Save the combined plot
# plt.tight_layout()
# plt.savefig(f"{save_dir}/combined_metrics_plot.png")
# plt.close()
