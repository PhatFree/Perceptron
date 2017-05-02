

def main():
    print("CS480:AI Homework 4, Problem 4 - Perceptron Training")
    raw_data = load_images()

    for image in raw_data:
        extract_features(image)

    print('done')


def load_images():
    with open('testdata.txt') as data_input:
        all_lines = data_input.read()
        images = all_lines.split('\n\n\n')  # divide images

        num_ones = 0
        num_fives = 0

        ones = []
        fives = []

        for i in range(len(images)):
            if len(images[i]) > 0:
                if images[i][0] == '1':
                    ones.append(images[i][1:])
                    num_ones += 1
                if images[i][0] == '5':
                    fives.append(images[i][1:])
                    num_fives += 1
            else:
                print('empty image')  # end-of-file

        print(num_ones, 'images of 1')
        print(num_fives, 'images of 5')
        print(num_ones + num_fives, 'images')

        # convert to matrix form
        one_arrays = [None]*num_ones
        five_arrays = [None]*num_fives
        for i in range(len(ones)):
            one_arrays[i] = [[int(x) for x in line.split()] for line in ones[i].split('\n')]
        for i in range(len(fives)):
            five_arrays[i] = [[int(x) for x in line.split()] for line in fives[i].split('\n')]

    print('images loaded')
    return one_arrays



def extract_features(image):
    tpose = list(zip(*image))  # transpose image for easier vertical analysis

    density = sum(sum(row) for row in image) / (sum(len(row) for row in image))

    symmetry_horz = 0
    symmetry_vert = 0

    #

    intersect_horz_min = 0

    intersect_horz_max = 0

    intersect_vert_min = 0
    intersect_vert_max = 0

    features = [density, symmetry_vert, symmetry_horz,
                intersect_horz_min, intersect_horz_max, intersect_vert_min, intersect_vert_max]
    return features


main()
