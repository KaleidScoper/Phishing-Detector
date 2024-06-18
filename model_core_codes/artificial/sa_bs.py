import pandas as pd
import random
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc


data = pd.read_csv("data48.csv")
X = data.iloc[:,optimal_features]
y = data.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

T = 100
T_min = 0.1
delta = 0.95
iters = 1000
best_ft = []
best_score = 0
best_metrics = 0
def model_metrics(model, x, y, pos_label=1):
    yhat = model.predict(x)
    yprob = model.predict_proba(x)[:, 1]
    fpr, tpr, _ = roc_curve(y, yprob, pos_label=pos_label)
    result = {'accuracy_score': accuracy_score(y, yhat),
              'f1_score_macro': f1_score(y, yhat, average="macro"),
              'precision': precision_score(y, yhat, average="macro"),
              'recall': recall_score(y, yhat, average="macro"),
              'auc': auc(fpr, tpr),
              'ks': max(abs(tpr - fpr))
              }
    return result


def bidirectional_selection(annealing=True,metrics='accuracy_score', verbose=True):

    for i in range(iters):
        # forward step
        excluded = list(set(X.columns) - set(included))
        random.shuffle(excluded)
        for new_column in excluded:
            dt.fit(x_train[included + [new_column]], y_train)
            latest_metrics = cross_val_score(RFC(),X[included + [new_column]],y,cv=10).mean()
            if latest_metrics > best_metrics:
                included.append(new_column)
                print('Add {} with metrics gain {:.6}'.format(new_column, latest_metrics - best_metrics))
                best_metrics = latest_metrics
                if best_metrics > best_score:
                    best_ft = included
                    best_score = best_metrics
            else :
                p = math.exp(-(latest_metrics - best_metrics) / T)
                r = np.random.uniform(low=0, high=1)
                if r < p:
                    included.append(new_column)
                    print('Annealing Add{} with metrics gain {:.6}'.format(new_column, latest_metrics - best_metrics))

        # backward step
        random.shuffle(included)
        for new_column in included:
            included.remove(new_column)
            dt.fit(x_train[included], y_train)

            latest_metrics = cross_val_score(RFC(), X[included], y, cv=10).mean()
            if latest_metrics < best_metrics:
                included.append(new_column)
            else:
                print('Drop{} with metrics gain {:.6}'.format(new_column, latest_metrics - best_metrics))
                best_metrics = latest_metrics
                if best_metrics > best_score:
                    best_ft = included
                    best_score = best_metrics


while T > T_min:
    bidirectional_selection(annealing=True,metrics='accuracy_score', verbose=True)
    T = T * delta
