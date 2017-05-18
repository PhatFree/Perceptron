"""
0 0 0 0 1
1 0 1 0 1
0 0 1 1 0
0 1 0 1 0
1 1 1 0 1
1 0 0 1 1
0 1 1 0 0
1 0 1 1 1
"""

import math
from functools import reduce

debug = True

original_data = [
    [0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1]]


class TreeNode:
    def __init__(self, leaf=True, prediction=0, feature=0, threshold=0.5):
        self.is_leaf = leaf
        self.prediction = prediction
        self.feature = feature
        self.threshold = threshold
        self.left_tree = self.right_tree = None

    def predict(self, data, debug=False):
        if self.is_leaf:
            return self.prediction
        if (debug):print('c{0}->'.format(self.feature), end="")
        if data[self.feature] < self.threshold:
            return self.left_tree.predict(data)
        return self.right_tree.predict(data)

    def display(self, array=None, index=1):
        if array is None: array = [0]
        array.extend(['x', 'x'])
        array[0] += 1
        if self.is_leaf:
            array[index] = ('p'+str(self.prediction))
        else:
            array[index] = ('c'+str(self.feature))
            self.left_tree.display(array, index*2)
            self.right_tree.display(array, index*2+1)
        return array


def entropy(label, total):
    return 0 if label == 0 else - (label / total) * math.log2(label / total)


def tree(data, used_classes=None):
    if used_classes is None:
        used_classes = [0] * (len(data[0]) - 1)
    if (debug):print('\ndata')
    if (debug):
        for row in data:
            print(row)
        if (debug):print(used_classes)
    classes = list(zip(*data))
    # last row is true_label

    num_labels = len(data)
    pos_label = sum(classes[-1])
    neg_label = num_labels - pos_label
    entropy_before = entropy(pos_label, num_labels) + entropy(neg_label, num_labels)
    if (debug):print('entropy:', entropy_before)

    if pos_label == num_labels or neg_label == num_labels:  # pure node
        if (debug):print('leaf node (pure label): prediction =', classes[-1][0])
        return TreeNode(prediction=classes[-1][0])

    if sum(used_classes) == len(used_classes):
        prediction = 1 if pos_label > neg_label else 0
        if (debug):print("no more usable features")
        if (debug):print("best prediction = ", prediction)
        return TreeNode(prediction=prediction)

    gains = [0]*len(used_classes)  # (yes_attr(label_yes, label_no), no_attr(label_yes, label_no))
    if (debug): print('entropy  positive\tnegative\tfeature')
    for i in range(len(used_classes)):
        if used_classes[i] == 1:
            continue
        feature = [[0, 0], [0, 0]]
        for j in range(len(classes[i])):
            feature[classes[i][j]][classes[-1][j]] += 1
        left, right = sum(feature[0]), sum(feature[1])  # with_attribute, without

        pos, neg = feature[0]
        entropy_noattr = entropy(pos, left) + entropy(neg, left)
        pos, neg = feature[1]
        entropy_attr = entropy(pos, right) + entropy(neg, right)

        entropy_feature = (left/(left+right))*entropy_attr + (right/(left+right))*entropy_noattr
        gains[i] = entropy_before - entropy_feature
        if (debug):print('class {0}: {1:8.6}\t{2:8.6}\t{3:8.6}'.format(i, entropy_attr, entropy_noattr, entropy_feature))

    if (debug):print('information gain:')
    if (debug):
        for gain in gains:
            print(gain)

    # find best feature to split on (highest information gain)
    # (index of best gain) == (class label to decide on at this node)
    best = reduce(lambda x, y: x if gains[x] >= gains[y] else y, range(len(gains)))

    used_classes[best] = 1
    if(debug):print('dropping class', best, 'entropy:', gains[best])
    if (debug):print(used_classes)

    left_data = [data[i] for i in range(len(data)) if classes[best][i] == 0]
    right_data = [data[i] for i in range(len(data)) if classes[best][i] == 1]

    node = TreeNode(leaf=False, feature=best)
    node.left_tree = tree(left_data, used_classes)
    node.right_tree = tree(right_data, used_classes)
    return node


def main():
    root = tree(original_data)
    print('\ntesting tree:')
    print('\tdata\t\tprediction')
    for row in original_data:
        print(row, end="  ")
        print(root.predict(row))
    print('tree:', root.display())


main()
