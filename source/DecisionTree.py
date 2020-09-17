import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, label=None, mean=None, parent=None, left=None, right=None):
        self.label = label
        self.mean = mean
        self.parent = parent
        self.left = left
        self.right = right
    def is_leaf(self):
        return self.left == None and self.right == None

class DecisionTree:
    def __init__(self):
        self.tree = Node()
        self.impurity_measure = None
        self.target = None

    def learn(self, X, y, impurity_measure='entropy'):
        self.impurity_measure = impurity_measure
        # x = [pd.DataFrame(x) for x in [X, y] if not isinstance(x, (pd.Series, pd.DataFrame))]
        #data = pd.concat([X_train, y_train], axis=1)
        data = pd.concat([X, y], axis=1)
        self.target = data.keys()[-1]
        self._recursive_build(data, self.tree)

    def predict(self, x):
        result = np.array([0]*len(x))
        for i, c in enumerate(x.index):
            result[i] = self._get_prediction(x.loc[c], self.tree)
        return result

    def _get_prediction(self, row, tree):
        if tree.is_leaf():
            return tree.label
        if row[tree.label] > tree.mean:
            return self._get_prediction(row, tree.right)
        else:
            return self._get_prediction(row, tree.left)

    def _recursive_build(self, data, node):
        if (data[self.target].values[0] ==  data[self.target].values).all():
            node.label = data[self.target].iloc[0]
            return
        col, mean = self._select_split_node(data)
        sub_data_1 = data[data[col] <= mean]
        sub_data_2 = data[data[col] > mean]
        node.label = col
        node.mean = mean
        if len(sub_data_1[self.target]) > 0:
            node.left = Node(parent=node)
            self._recursive_build(sub_data_1, node.left)
        if len(sub_data_2[self.target]) > 0:
            node.right = Node(parent=node)
            self._recursive_build(sub_data_2, node.right)

    def _select_split_node(self, data):
        if self.impurity_measure == 'entropy':
            ig = self._info_gain(data)
            opt_ig, opt_col, opt_mean = 0, 0, 0
            for k in ig.keys():
                if ig[k]['ig'] > opt_ig:
                    opt_ig = ig[k]['ig']
                    opt_col = k
                    opt_mean = ig[k]['mean']
        return opt_col, opt_mean

    def _info_gain(self, data):
        h_full = 0
        target_val, count = np.unique(data[self.target].values, return_counts=True)
        for c in count:
            frac = c/len(data[self.target].values)
            h_full += -frac*np.log2(frac)   # Calculate entropy for entire dataset
        h_att = {k: self._entropy(data, k) for k in data.keys()[:-1]}  # Calculate entropy for each attribute
        return {k: {'ig': (h_full - h_att[k][0]), 'mean': h_att[k][1]} for k in h_att}

    def _entropy(self, data, attribute):
        zero = np.finfo(float).eps
        mean = np.mean(data[attribute].values)
        entropy_attribute = 0
        sub_data = [data[data[attribute] <= mean], data[data[attribute] > mean]]
        for subset in sub_data:
            entropy_set = 0
            for target_val in np.unique(subset[self.target].values):
                n = len(subset[subset[self.target] == target_val])
                d = len(subset)
                frac = n/(d+zero)
                entropy_set += -frac*np.log2(frac+zero)
            entropy_attribute += -(d/len(data))*entropy_set
        return abs(entropy_attribute), mean