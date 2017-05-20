from nnetwork import *
from functools import reduce


def main():
    print("CS480:AI Homework 5, Problem 3 - Neural Network Training")
    images, true_classes = load_images()

    train = []
    train_labels = []
    test = []
    test_labels = []

    for i in range(len(images)):
        feature = extract_features(images[i])
        if random.randint(0, 10) < 9:
            train.append(feature)
            exp = [0]*10
            exp[true_classes[i]] = 1
            train_labels.append(exp)
        else:
            test.append(feature)
            test_labels.append(true_classes[i])
    print('features extracted')

    epochs = 10
    learn_rate = 0.01

    for h in range(10, 20, 10):
        network = new_network(49, h, 10)
        score = train_network(network, train, train_labels, test, test_labels, learn_rate, epochs)

        print('testing accuracy: {0:.4}%'.format(score))

    print('done')


def load_images():
    with open('testdata.txt') as data_input:
        all_lines = data_input.read()
        images = all_lines.split('\n\n\n')  # divide images

        class_counts = [0] * 10
        img_strings = []  # raw image
        img_classes = []  # expected

        for i in range(len(images)):
            if len(images[i]) > 0:

                img_strings.append(images[i][1:])

                num = ord(images[i][0]) - ord('0')
                class_counts[num] += 1
                img_classes.append(num)
            # else should be at end-of-file

        # convert to matrix form
        raw_images = [None]*(sum(class_counts))

        for i in range(len(img_strings)):
            raw_images[i] = [[int(x) for x in line.split()] for line in img_strings[i].split('\n')]

    print(sum(class_counts), 'images loaded', class_counts)
    return raw_images, img_classes


def extract_features(image):
    # divide image into 4x4 regions
    # one feature per region, the average pixel value

    features = []
    rows, cols = len(image) // 4, len(image[0]) // 4

    for r in range(rows):
        for c in range(cols):
            sum = 0
            for i in range(4):
                for j in range(4):
                    sum += image[r*4 + i][c*4 + j]
            avg = sum / (4*4)
            features.append(avg)

    return features


main()
