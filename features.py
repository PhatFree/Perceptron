import Perceptron
import random


def main():
    print("CS480:AI Homework 4, Problem 4 - Perceptron Training")
    images, true_classes = load_images()

    train = []
    test = []
    for i in range(len(images)):
        if true_classes[i] == 1:
            feature = extract_features(images[i], 1)
        elif true_classes[i] == 5:
            feature = extract_features(images[i], 0)
        if random.randint(0, 10) < 8:
            train.append(feature)
        else:
            test.append(feature)
        # print(feature)
    print('features extracted')

    """
    we'll have to create two separate perceptrons, one for 1's and one for fives.
    they'll have to be passed a 2d array, with rows being "images" and with cols
    being the values of the features, and the expected will be feature[-1]
    """

    weights = Perceptron.train_weights(train, 0.1, 100)
    print('perceptron trained')

    correct = 0
    for i in range(len(test)):
        prediction = Perceptron.predict(test[i], weights)
        if prediction == test[i][-1]:
            correct += 1
    print('accuracy: {0:.4}%'.format(correct * 100 / len(test)))

    print('done')


def load_images():
    with open('testdata.txt') as data_input:
        all_lines = data_input.read()
        images = all_lines.split('\n\n\n')  # divide images

        num_ones = 0
        num_fives = 0

        img_strings = []  # raw image
        img_classes = []  # expected

        for i in range(len(images)):
            if len(images[i]) > 0:

                img_strings.append(images[i][1:])

                if images[i][0] == '1':
                    num_ones += 1
                    img_classes.append(1)

                if images[i][0] == '5':
                    num_fives += 1
                    img_classes.append(5)
            # else should be at end-of-file

        # convert to matrix form
        raw_images = [None]*(num_fives+num_ones)  # preallocate space

        for i in range(len(img_strings)):
            raw_images[i] = [[int(x) for x in line.split()] for line in img_strings[i].split('\n')]

    print(num_ones + num_fives, 'images loaded')
    return raw_images, img_classes


def extract_features(image, expected):
    height, width = len(image), len(image[0])
    tpose = list(zip(*image))  # transpose image for easier vertical analysis

    density = sum(sum(row) for row in image) / (height*width)

    matches = 0
    for irow in range(height//2):
        for col in range(width):
            if image[irow][col] == image[irow+height//2][col]:
                matches += 1
    symmetry_horz = matches / (height*width)  # symmetry across horizontal axis

    matches = 0
    # height in original becomes width in transpose
    for irow in range(width//2):
        for col in range(height):
            if tpose[irow][col] == tpose[irow+width//2][col]:
                matches += 1
    symmetry_vert = matches / (height*width)  # symmetry across vertical axis

    intersect_horz_min = height  # minimum count of intersections in any row
    intersect_horz_max = 0  # maximum count of intersections in any row
    for row in image:
        intersections = 0
        for i in range(1, len(row)):
            if row[i-1] == 1 and row[i] == 0:
                intersections += 1
        intersect_horz_min = min(intersect_horz_min, intersections)
        intersect_horz_max = max(intersect_horz_max, intersections)

    intersect_vert_min = 0  # minimum count of intersections in any column
    intersect_vert_max = 0  # maximum count of intersections in any column
    for row in tpose:
        intersections = 0
        for i in range(1, len(row)):
            if row[i-1] == 1 and row[i] == 0:
                intersections += 1
        intersect_vert_min = min(intersect_vert_min, intersections)
        intersect_vert_max = max(intersect_vert_max, intersections)

    features = [density, symmetry_vert, symmetry_horz,
                intersect_horz_min, intersect_horz_max, intersect_vert_min, intersect_vert_max, expected]
    return features


main()
