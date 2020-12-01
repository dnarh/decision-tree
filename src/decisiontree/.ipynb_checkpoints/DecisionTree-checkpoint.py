"""
Module containing the DecisionTree class for building decision trees.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Node:
    """
    Class for constructing nodes in a decision tree.

    Attributes:
        is_leaf: Returns True if the node object is a leaf node in a decision tree, False if not
    """

    def __init__(self, label=None, mean=None, parent=None, left=None, right=None):
        """
        Inits Node class.

        Args:
            label:
        """
        self.label = label
        self.mean = mean
        self.parent = parent
        self.left = left
        self.right = right
        self.majority = None

    def is_leaf(self):
        """
        Returns True if the node object is a leaf node in a decision tree, False if not
        """
        return self.left == None and self.right == None


class DecisionTree:
    """
    Class for constructing decision trees.

    Attributes:
        learn: Builds a decision tree classifier from input training feature and label data.
        fit: Predicts the label of a feature input, using the constructed decision tree.
    """

    def __init__(self):
        """
        Inits DecisionTree class.
        """

        self.tree = Node()
        self.impurity_measure = None
        self.target = None

    def learn(self, X_train, y_train, impurity_measure='entropy', prune=False):
        """
        Builds a decision tree classifier from input training feature and label data.

        Args:
            X_train: feature data used for training
            y_train: label data used for training
            impurity_measure: method used for calculating impurity of the features. Defaults to 'entropy'.
            prune: sets error-reduced pruning to be used or not. Defaults to False.

        Returns:
            DecisionTree object.
        """

        self.impurity_measure = impurity_measure
        if prune:
            X_train, X_prune, y_train, y_prune = train_test_split(X_train, y_train, test_size=0.3, random_state=1)
        data_train = pd.concat([X_train, y_train], axis=1)
        self.target = data_train.keys()[-1]
        self._recursive_build(data_train, self.tree)
        if prune:
            self._prune(X_prune, y_prune, self.tree)
        return self

    def predict(self, x):
        """
        Predicts the label of a feature input, using the constructed decision tree.

        Args:
            x: Input feature(s)

        Returns:
            result: Array of prediction results.
        """

        result = np.array([0]*len(x))
        for i, c in enumerate(x.index):
            result[i] = self._get_prediction(x.loc[c], self.tree)
        return result

    def _prune(self, x_prune, y_prune, tree):
        """
        Perform error-reduced pruning on decision tree recursively.

        Args:
            x_prune: Sub set of features used for pruning.
            y_prune: Sub set of labels used for pruning.
            tree: Decision tree, object of class Node.

        Returns:
            Error for the node in question.
        """

        if tree.is_leaf():
            return len(y_prune) - len(y_prune[y_prune == tree.label])
        x_left, x_right, y_left, y_right = self._split_prune_data(x_prune, y_prune)
        e_left = self._prune(x_left, y_left, tree.left)
        e_right = self._prune(x_right, y_right, tree.right)
        e_majority = len(y_prune) - len(y_prune[y_prune == tree.majority])
        if e_majority <= e_left + e_right:
            tree.label = tree.majority
            tree.left = tree.right = None
            return e_majority
        return e_left + e_right

    def _split_prune_data(self, x, y):
        """"
        Split prune data set by mean of the node split feature values.

        Args:
            x: data set features
            y: data set labels

        Returns:
            x_left: features for left node
            x_right: features for right node
            y_left: labels for left node
            y_right: labels for right node
        """

        mean = np.mean(x[self.tree.label])
        mask = x[self.tree.label] <= mean
        x_left = x[mask]
        y_left = y[mask]
        x_right = x[~mask]
        y_right = y[~mask]
        return x_left, x_right, y_left, y_right

    def _get_prediction(self, row, tree):
        """
        Recurse through decision tree to get the correct prediction for input features in row.

        Args:
            row: Input features.
            tree: Decision tree.

        Returns:
            Node label/prediction if leaf node is reached in tree, else calls itself recursively with the
            child node (node.left/node.right) as inputs.
        """

        if tree.is_leaf():
            return tree.label
        if row[tree.label] > tree.mean:
            return self._get_prediction(row, tree.right)
        else:
            return self._get_prediction(row, tree.left)

    def _recursive_build(self, data, node):
        """
        Build decision tree recursively until all labels for the outer-most nodes (leaf node) all have
        equal labels.

        Args:
            data: combined DataFrame of feature and label data
            node: current decision tree node, object of class Node
        """

        if (data[self.target].values[0] == data[self.target].values).all():
            node.label = data[self.target].iloc[0]
            return
        node.majority = 0 if len(data[data[self.target] == 0]) >= len(data[data[self.target] == 1]) else 1
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
        """
        Calls functions for selecting feature to split data on and calculating its mean, based on
        the impurity measure selected for the tree.

        Args:
            data: combined DataFrame of feature and label data

        Returns:
            opt_col: optimal feature to split data on
            opt_mean: mean of opt_col
        """

        if self.impurity_measure == 'entropy':
            ig = self._info_gain(data)
            opt_ig, opt_col, opt_mean = 0, 0, 0
            for k in ig.keys():
                if ig[k]['ig'] > opt_ig:
                    opt_ig = ig[k]['ig']
                    opt_col = k
                    opt_mean = ig[k]['mean']
        elif self.impurity_measure == 'gini':
            g = self._gini_index(data)
            opt_gini, opt_col, opt_mean = 1, 0, 0
            for k in g.keys():
                if g[k]['gini_idx'] < opt_gini:
                    opt_gini = g[k]['gini_idx']
                    opt_col = k
                    opt_mean = g[k]['mean']
        return opt_col, opt_mean

    def _info_gain(self, data):
        """
        Calculates the information gain for each feature in the input data.

        Args:
            data: combined DataFrame of feature and label data

        Returns:
            Dictionary containing the information gain and mean value for each feature.
        """

        h_full = 0
        target_val, count = np.unique(data[self.target].values, return_counts=True)
        for c in count:
            frac = c/len(data[self.target].values)
            h_full += -frac*np.log2(frac)   # Calculate entropy for entire dataset
        h_feat = {f: self._entropy(data, f) for f in data.keys()[:-1]}  # Calculate entropy for each feature
        return {k: {'ig': (h_full - h_feat[k][0]), 'mean': h_feat[k][1]} for k in h_feat}

    def _entropy(self, data, feature):
        """
        Calculates entropy for the input feature.

        Args:
            data: combined DataFrame of feature and label data
            feature: feature column name in the data DataFrame

        Returns:
            entropy_feature: summed entropy for feature
            mean: mean of the feature values
        """
        zero = np.finfo(float).eps
        mean = np.mean(data[feature].values)
        entropy_feature = 0
        sub_data = [data[data[feature] <= mean], data[data[feature] > mean]]
        for subset in sub_data:
            entropy_set = 0
            for target_val in np.unique(subset[self.target].values):
                n = len(subset[subset[self.target] == target_val])
                d = len(subset)
                frac = n/(d+zero)
                entropy_set += -frac*np.log2(frac+zero)
            entropy_feature += -(d/len(data))*entropy_set
        return abs(entropy_feature), mean

    def _gini_index(self, data):
        """
        Calculates the gini index for each feature in the input data.

        Args:
            data: combined DataFrame of feature and label data

        Returns:
            Dictionary containing gini index and mean value for each feature.
        """

        g = dict()
        zero = np.finfo(float).eps
        for attribute in data.keys()[:-1]:
            g_attribute = 0
            mean = np.mean(data[attribute].values)
            sub_data = [data[data[attribute] <= mean], data[data[attribute] > mean]]
            for subset in sub_data:
                g_set = 0
                for target_val in np.unique(subset[self.target].values):
                    n = len(subset[subset[self.target] == target_val])
                    d = len(subset)
                    frac = n / (d + zero)
                    g_set += np.float_power(frac, 2)
                g_attribute += (d/len(data))*(1-g_set)
            g[attribute] = (g_attribute, mean)
        return {k: {'gini_idx': g[k][0], 'mean': g[k][1]} for k in g}