import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import  url_cut, get_char_ngrams, load_data
import numpy as np
from vocab import Vocab, read_vocab
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import warnings
import joblib
from sklearn.model_selection import train_test_split


vocab = read_vocab('./vocab.txt')
data0_path="./data/legtimate-58w.txt"
data1_path="./data/phish-58w.txt"
train_prop=0.8
val_prop=0.1
test_prop=0.1


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

def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets

def get_part_feature(data):
    urls_char=[]
    for url in data:
        urls_char.append(get_char_ngrams(url))
    data = url_jieduan(urls_char, 120)
    data = [(vocab.convert_tokens_to_ids(sentence), 1)
            for sentence in data]
    inputs = DataLoader(data, batch_size=256, collate_fn=collate_fn, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load("./model_storage/model-cc.pkl")
    flag = True
    for batch in inputs:
        x, y = [x.to(device) for x in batch]
        out = model(x)
        feature = model.features
        array_batch_i = feature.cpu().detach().numpy()
        if flag:
            a = array_batch_i
        else:
            a = np.concatenate((a, array_batch_i))
        flag = False
    return a


urls0 = load_data(data0_path)
urls0_cnn = pd.DataFrame(get_part_feature(urls0))
urls0_art = urls_artificial(data0_path)
data0 = pd.concat([urls0_art,urls0_cnn],axis=1,ignore_index=True)
data0['Target'] = 0
X0_test = data0[:(int)(test_prop * len(data0)):]
X0_train = data0[(int)(test_prop * len(data0)):]


urls1 = load_data(data1_path)
urls1_cnn = pd.DataFrame(get_part_feature(urls1))
urls1_art = urls_artificial(data1_path)
data1 = pd.concat([urls1_art,urls1_cnn],axis=1,ignore_index=True)
data1['Target'] = 1
X1_test = urls1[:(int)(test_prop * len(data1))]
X1_train = urls1[(int)(test_prop * len(data1)):]

X_train = pd.concat([X0_train,X1_train],axis=0,ignore_index=True).iloc[:,1:-1]
X_test = pd.concat([X0_test,X1_test],axis=0,ignore_index=True).iloc[:,1:-1]
y_train = pd.concat([X0_train,X1_train],axis=0,ignore_index=True).iloc[:,-1]
y_test = pd.concat([X0_test,X1_test],axis=0,ignore_index=True).iloc[:,-1]

rfc = RFC()
rfc.fit(X_train, y_train)

y_pre = rfc.predict(X_test)
print(classification_report(y_test, y_pre, target_names=['legtimate', 'phish'], digits=5))