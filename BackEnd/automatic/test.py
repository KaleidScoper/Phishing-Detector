import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import load_sentence_polarity
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from tqdm.auto import tqdm

embedding_dim = 128
num_class = 2
batch_size = 256


def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets


class CnnDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, 3)
        self.conv2 = nn.Conv1d(embedding_dim, 64, 5)
        self.linear = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.embedding(x)

        x1 = F.relu(self.conv1(x.permute(0, 2, 1)))
        x2 = F.relu(self.conv2(x.permute(0, 2, 1)))

        pool1 = F.max_pool1d(x1, kernel_size=x1.shape[2])
        pool2 = F.max_pool1d(x2, kernel_size=x2.shape[2])

        x1 = pool1.squeeze(dim=2)
        x2 = pool2.squeeze(dim=2)
        x = torch.cat([x1, x2], dim=1)
        self.features = x

        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        return out


train_data, val_data, test_data, vocab = load_sentence_polarity(100)

train_dataset = CnnDataset(train_data)
val_dataset = CnnDataset(val_data)
test_dataset = CnnDataset(test_data)

train_data_loader = DataLoader(train_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("./model_storage/model-cc.pkl")
model.eval()

TP=FP=TN=FN=0
for batch in tqdm(test_data_loader, desc=f"Testing"):
    inputs, targets = [x.to(device) for x in batch]
    log_probs = model(inputs)

    pre = log_probs.argmax(dim=1).cpu().numpy()[0]
    label = targets.cpu().numpy()[0]
    if pre == 1 and label == 1:
        TP += 1
    if pre == 1 and label == 0:
        FP += 1
    if pre == 0 and label == 1:
        FN += 1
    if pre == 0 and label == 0:
        TN += 1

acc = (TP + TN) / (TP + FP + TN + FN)
pre = TP / (TP + FP)
rec = TP / (TP + FN)
f1s = (2 * pre * rec) / (pre + rec)