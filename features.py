import Perceptron


def main():
    print("CS480:AI Homework 4, Problem 4 - Perceptron Training")
    images, true_classes = load_images()

    print(true_classes)
    raw = []
    for i in range(len(images)):
        if true_classes[i] == 5:
            feature = extract_features(images[i], 0.0)
        elif true_classes[i] == 1:
            feature = extract_features(images[i], 1.0)

        raw.append(feature)
        print(feature)

    """
    we'll have to create two separate perceptrons, one for 1's and one for fives.
    they'll have to be passed a 2d array, with rows being "images" and with cols
    being the values of the features, and the expected will be feature[-1]
    """

    # Perceptron.train_weights(raw, 0.1, len(raw))

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
            else:
                print('empty image')  # should be at end-of-file

        print(num_ones, 'images of 1')
        print(num_fives, 'images of 5')
        print(num_ones + num_fives, 'images')

        # convert to matrix form
        raw_images = [None]*(num_fives+num_ones)  # preallocate space

        for i in range(len(img_strings)):
            raw_images[i] = [[int(x) for x in line.split()] for line in img_strings[i].split('\n')]

    print('images loaded')
    return raw_images, img_classes


def extract_features(image, expected):
    height, width = len(image), len(image[0])

    tpose = list(zip(*image))  # transpose image for easier vertical analysis

    density = sum(sum(row) for row in image) / (height*width)

    x = 0
    for irow in range(height//2):
        for col in range(len(image[irow])):
            if image[irow][col] == image[irow+height//2][col]:
                x += 1
    symmetry_horz = x / (height*width)  # symmetry across horizontal axis

    x = 0
    for irow in range(height//2):
        for col in range(len(tpose[irow])):
            if tpose[irow][col] == tpose[irow+height//2][col]:
                x += 1
    symmetry_vert = x / (height*width)  # symmetry across vertical axis


    #

    intersect_horz_min = 0.0

    intersect_horz_max = 0.0

    intersect_vert_min = 0.0
    intersect_vert_max = 0.0

    features = [density, symmetry_vert, symmetry_horz,
                intersect_horz_min, intersect_horz_max, intersect_vert_min, intersect_vert_max, expected]
    return features


main()
