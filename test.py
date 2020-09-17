from source.DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import pandas as pd

def implement_decision_tree():
    data = pd.read_csv(os.path.join('data', 'data_banknote_authentication.csv'), header=None)

    X = data[data.keys()[:-1]]
    y = data[data.keys()[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    dt = DecisionTree()
    dt.learn(X_train, y_train, impurity_measure='entropy')
    predictions = pd.Series(dt.predict(X_test))

    true_pos = 0
    for c, val in enumerate(predictions):
        if val == y_test.iloc[c]:
            true_pos += 1
    perc = true_pos/len(predictions)

    print('Prediction accuracy: {:.2f}''%'''.format(perc*100))

if __name__ == '__main__':
    implement_decision_tree()
