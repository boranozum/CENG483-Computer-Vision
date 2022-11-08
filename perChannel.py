import utils
from matplotlib.image import imread

file = open('InstanceNames.txt')
instances = file.readlines()

histType = int(input('Histogram Type (0- Per channel, 1-3D): '))

interval = int(input('How many bins?: '))

gridInput = input('Grid based extraction? (Y/N): ')
gridN = 1

if gridInput == 'Y':
    gridN = int(input('Enter the grid size N (2,4,6,8): '))

if histType == 0:

    rHist = []
    gHist = []
    bHist = []

    for instance in instances:
        filename = 'support_96/' + instance.strip()
        image = imread(filename)

        grid = utils.divideIntoGrid(image, gridN)

        cellR = []
        cellG = []
        cellB = []

        for cell in grid:

            hist = utils.perChannelHistogram(cell, interval)
            cellR.append(utils.l1Normalizer(hist[0]))
            cellG.append(utils.l1Normalizer(hist[1]))
            cellB.append(utils.l1Normalizer(hist[2]))

        rHist.append(cellR)
        gHist.append(cellG)
        bHist.append(cellB)

    # Query 1

    for j in range(1, 4):

        totalCorrect = 0

        for instance in instances:
            filename = 'query_' + str(j) + '/' + instance.strip()
            image = imread(filename)

            queryGrid = utils.divideIntoGrid(image, gridN)

            cellR = []
            cellG = []
            cellB = []

            for cell in queryGrid:
                hist = utils.perChannelHistogram(cell, interval)
                cellR.append(utils.l1Normalizer(hist[0]))
                cellG.append(utils.l1Normalizer(hist[1]))
                cellB.append(utils.l1Normalizer(hist[2]))

            lowestAt = -1
            minDivergence = 99999

            for i in range(len(instances)):
                div = 0
                for k in range(gridN*gridN):

                    divR = utils.JSDivergence(cellR[k], rHist[i][k])
                    divG = utils.JSDivergence(cellG[k], gHist[i][k])
                    divB = utils.JSDivergence(cellB[k], bHist[i][k])

                    div += (divR + divG + divB) / 3

                div /= (gridN*gridN)

                if div < minDivergence:
                    minDivergence = div
                    lowestAt = i

            if instance == instances[lowestAt]:
                totalCorrect += 1

        top1Acc = totalCorrect / len(instances)

        print(f'Top-1 accuracy of the query-{j} dataset is: {top1Acc}')
else:
    histograms = []
    for instance in instances:
        filename = 'support_96/' + instance.strip()
        image = imread(filename)

        grid = utils.divideIntoGrid(image, gridN)
        histPerCell = []

        for cell in grid:
            hist = utils.threeDHistogram(cell,interval)
            histPerCell.append(utils.l1Normalizer(hist))

        histograms.append(histPerCell)

    # Query 1
    for i in range(1,4):
        totalCorrect = 0

        for instance in instances:
            filename = 'query_' + str(i) + '/' + instance.strip()
            image = imread(filename)

            queryGrid = utils.divideIntoGrid(image, gridN)
            queryHist = []

            for cell in queryGrid:
                h = utils.threeDHistogram(image, interval)
                queryHist.append(utils.l1Normalizer(h))

            lowestAt = -1
            minDivergence = 999999

            for j in range(len(instances)):
                div = 0
                for k in range(gridN*gridN):
                    div += utils.JSDivergence(queryHist[k], histograms[j][k])

                div /= (gridN*gridN)

                if div < minDivergence:
                    minDivergence = div
                    lowestAt = j

            if instance == instances[lowestAt]:
                totalCorrect += 1

        top1Acc = totalCorrect / len(instances)
        print(f'Top-1 accuracy of the query-{i} dataset is: {top1Acc}')

