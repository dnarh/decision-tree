from src.DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
import time
import pandas as pd


def accuracy(y_pred, y_val):
    true_pos = 0
    for c, val in enumerate(y_pred):
        if val == y_val.iloc[c]:
            true_pos += 1
    return (true_pos / len(y_pred))*100


def run():
    data = pd.read_csv(os.path.join('data', 'data_banknote_authentication.csv'), header=None)

    X = data[data.keys()[:-1]]
    y = data[data.keys()[-1]]

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    for imp_meas in ['entropy', 'gini']:
        for pr in [True, False]:
            tic = time.perf_counter()
            dt = DecisionTree()
            dt.learn(X, y, impurity_measure=imp_meas, prune=pr)
            pred = pd.Series(dt.predict(X_test))
            toc = time.perf_counter()

            print('\nImpurity measure: {}, Error-reduced pruning: {}'.format(imp_meas, pr))
            print('Time consumption: {:0.4f} seconds'.format((toc-tic)))
            print('Accuracy: {:.2f}%'.format(accuracy(pred, y_test)))
            if not pr:
                cmp = compare(X, y, X_test, imp_meas)
                print('sklearn accuracy: {:.2f}%'.format(accuracy(cmp, y_test)))

def compare(X, y, X_test, imp_meas):
    sk_dt = DecisionTreeClassifier(criterion=imp_meas)
    sk_dt.fit(X,y)
    return pd.Series(sk_dt.predict(X_test))


if __name__ == '__main__':
    run()
