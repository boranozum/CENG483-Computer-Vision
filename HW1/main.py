from matplotlib.image import imread
import numpy as np

def perChannelHistogram(image, interval):
    '''

    Returns a 2D numpy array where each column represents a color channel with a size of given interval

    :param image:
    :param interval: How many bins will be required?
    :return: numpy array
    '''
    imageHeight = image.shape[0]

    size = 256 / interval

    result = np.zeros([3, interval], int)

    for i in range(imageHeight):

        for j in range(interval):
            result[0, j] += ((image[i][:, 0] >= size * j) & (image[i][:, 0] < size * (j + 1))).sum()
            result[1, j] += ((image[i][:, 1] >= size * j) & (image[i][:, 1] < size * (j + 1))).sum()
            result[2, j] += ((image[i][:, 2] >= size * j) & (image[i][:, 2] < size * (j + 1))).sum()

    return result


def threeDHistogram(image, bins):
    '''

    Returns a 1D array with a size of given bin count. Uses broadcasting features to extract color values and count
    the occurrences of each value.

    :param image:
    :param bins: How many bins will be required?
    :return: Numpy array
    '''

    imageHeight = image.shape[0]

    res = np.zeros(bins**3,dtype=int)

    interval = int(256/bins)

    tempImage = image.reshape(imageHeight*imageHeight,3)

    b = np.floor_divide(tempImage, interval)
    c = b[:, 0] * (bins**2) + b[:, 1] * bins + b[:, 2]

    unique, counts = np.unique(c, return_counts=True)

    res[unique] = counts

    return res

def l1Normalizer(hist):
    '''

    Takes a histogram and apply l1 normalization on each element.

    :param hist:
    :return: Numpy array
    '''
    norm = np.linalg.norm(hist,ord=1)

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
    '''

    Returns nxn numpy array

    :param image:
    :param n: Grid dimension
    :return: Numpy array
    '''
    grid = []

    size = int(96 / n)

    for i in range(n):
        for j in range(n):
            cell = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
            grid.append(cell)

    return grid


file = open('../InstanceNames.txt')
instances = file.readlines()

histType = int(input('Histogram Type (0- Per channel, 1-3D): '))

interval = int(input('How many bins?: '))

gridInput = input('Grid based extraction? (Y/N): ')
gridN = 1

if gridInput == 'Y':
    gridN = int(input('Enter the grid size N (2,4,6,8): '))

if histType == 0:           # Per Channel histogram selected

    rHist = []
    gHist = []
    bHist = []

    # Each instance in support set has been processed and for each of the cells in grid, 3 histogram is extracted
    for instance in instances:
        filename = 'support_96/' + instance.strip()
        image = imread(filename)

        grid = divideIntoGrid(image, gridN)

        cellR = []
        cellG = []
        cellB = []

        # Extract histograms for every cell
        for cell in grid:

            hist = perChannelHistogram(cell, interval)
            cellR.append(l1Normalizer(hist[0]))
            cellG.append(l1Normalizer(hist[1]))
            cellB.append(l1Normalizer(hist[2]))

        rHist.append(cellR)
        gHist.append(cellG)
        bHist.append(cellB)

    # Each query set is processed
    for j in range(1, 4):

        totalCorrect = 0

        # Each instance in query sets has been processed and for each of the cells in grid, 3 histogram is extracted
        for instance in instances:
            filename = 'query_' + str(j) + '/' + instance.strip()
            image = imread(filename)

            queryGrid = divideIntoGrid(image, gridN)

            cellR = []
            cellG = []
            cellB = []

            # Extract histograms for every cell
            for cell in queryGrid:
                hist = perChannelHistogram(cell, interval)
                cellR.append(l1Normalizer(hist[0]))
                cellG.append(l1Normalizer(hist[1]))
                cellB.append(l1Normalizer(hist[2]))

            # Min divergence value and the index of which the minimum has reached
            lowestAt = -1
            minDivergence = 99999

            for i in range(len(instances)):
                div = 0

                # JSD divergence for each corresponding cell is computed channel by channel
                for k in range(gridN*gridN):

                    divR = JSDivergence(cellR[k], rHist[i][k])
                    divG = JSDivergence(cellG[k], gHist[i][k])
                    divB = JSDivergence(cellB[k], bHist[i][k])

                    div += (divR + divG + divB) / 3     # Average of channel divergences

                div /= (gridN*gridN)    # Average of the cells

                if div < minDivergence:
                    minDivergence = div
                    lowestAt = i

            # Check whether minimum divergence is found with correct image in the query set
            if instance == instances[lowestAt]:
                totalCorrect += 1

        # Calculate the accuracy
        top1Acc = totalCorrect / len(instances)

        print(f'Top-1 accuracy of the query-{j} dataset is: {top1Acc}')
else:
    histograms = []

    # Each instance in query sets has been processed and for each of the cells in grid, 1 1D histogram is extracted
    for instance in instances:
        filename = 'support_96/' + instance.strip()
        image = imread(filename)

        grid = divideIntoGrid(image, gridN)
        histPerCell = []

        # Extract histograms for every cell
        for cell in grid:
            hist = threeDHistogram(cell,interval)
            histPerCell.append(l1Normalizer(hist))

        histograms.append(histPerCell)

    # Each query set is processed
    for i in range(1,4):
        totalCorrect = 0

        # Each instance in query sets has been processed and for each of the cells in grid, 1 histogram is extracted
        for instance in instances:
            filename = 'query_' + str(i) + '/' + instance.strip()
            image = imread(filename)

            queryGrid = divideIntoGrid(image, gridN)
            queryHist = []

            # Extract histograms for every cell
            for cell in queryGrid:
                h = threeDHistogram(image, interval)
                queryHist.append(l1Normalizer(h))

            # Min divergence value and the index of which the minimum has reached
            lowestAt = -1
            minDivergence = 999999

            for j in range(len(instances)):
                div = 0
                # JSD divergence for each corresponding cell is computed
                for k in range(gridN*gridN):
                    div += JSDivergence(queryHist[k], histograms[j][k])

                div /= (gridN*gridN)    # Average of the cells

                if div < minDivergence:
                    minDivergence = div
                    lowestAt = j

            # Check whether minimum divergence is found with correct image in the query set
            if instance == instances[lowestAt]:
                totalCorrect += 1

        # Calculate the accuracy
        top1Acc = totalCorrect / len(instances)
        
        print(f'Top-1 accuracy of the query-{i} dataset is: {top1Acc}')

