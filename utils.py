import numpy as np


def perChannelHistogram(image, interval):
    imageHeight = image.shape[0]

    size = 256 / interval

    result = np.zeros([3, interval], int)

    for i in range(imageHeight):

        for j in range(interval):
            result[0, j] += ((image[i][:, 0] >= size * j) & (image[i][:, 0] < size * (j + 1))).sum()
            result[1, j] += ((image[i][:, 1] >= size * j) & (image[i][:, 1] < size * (j + 1))).sum()
            result[2, j] += ((image[i][:, 2] >= size * j) & (image[i][:, 2] < size * (j + 1))).sum()

    return result


def threeDHistogram(image, interval):
    imageHeight = image.shape[0]

    result = np.zeros(interval, int)

    size = 256 / interval

    for i in range(imageHeight):
        for j in range(interval):
            a = (image >= size * j) & (image < size * (j + 1))
            result[j] = ((a[0][:, 0] == True) & (a[0][:, 1] == True) & (a[0][:, 2] == True)).sum()

    return result

def l1Normalizer(hist):
    norm = np.linalg.norm(hist)

    if norm == 0:
        norm = 0.00001

    res = hist / norm

    res[np.where(res == 0)] = 0.00001

    return res


def KLDivergence(query, search):
    sumArray = query * np.log(query / search)
    return sumArray.sum()


def JSDivergence(query, search):
    lhs = 0.5 * KLDivergence(query, (query + search) / 2)
    rhs = 0.5 * KLDivergence(search, (query + search) / 2)

    return lhs + rhs


def divideIntoGrid(image, n):
    grid = []

    size = int(96 / n)

    for i in range(n):
        for j in range(n):
            cell = image[i * size:, :(i + 1) * size][j * size:(j + 1) * size]
            grid.append(cell)

    return grid
