# inputs will only contain 1's or 0's
# weights will contain inputs+1, the extra weight is for the threshold


# weight[0] = threshold
# weight[1] = density
# weight[2] = symmetry
# weight[3] = maxium number of line intersections
sampleweights = [-.1, .2, .3, -.4]
# dataset[i][0] = density
# dataset[i][1] = symmetry
# dataset[i][2] = maximum number of line intersections
# dataset[i][-1/3] = expected
sampledataset = [[.4, .2, .3, 0], [1, .53, .99, 1], [.982, 3.1, .23, 1], [.33, 1, 2, 0]]


def predict(inputs, weights):
    # weights[0] is the threshold
    activation = weights[0]
    # print("weights:" + str(weights))
    # print("inputs:" + str(inputs))
    # Sum wi * Ii
    for i in range(len(inputs) - 1):
        activation += weights[i + 1] * inputs[i]
    return 1.0 if activation > 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
    weights = [0.0] * len(train[0])
    # epoch = times to train
    best_weights = weights
    best_error = -1
    for epoch in range(n_epoch):
        sum_error = 0.0
        # begin training

        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        if best_error == -1 or best_error > sum_error:
            best_weights = weights
        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return best_weights


"""
def main():
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]
    l_rate = 0.1
    n_epoch = 10
    weights = train_weights(dataset, l_rate, n_epoch)
    print(weights)

    # for row in sampledataset:
    #   prediction = predict(row,sampleweights)
    #  print("Expected=%d, Predicted=%d" % (row[-1],prediction))


main()
"""
