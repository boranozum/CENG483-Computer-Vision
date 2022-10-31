import utils
from matplotlib.image import imread
import numpy as np

file = open('InstanceNames.txt')
instances = file.readlines()

interval = int(input('Bin size: '))

gridInput = input('Grid based extraction? (Y/N):')
gridN = 1

rHist = []
gHist = []
bHist = []

if gridInput == 'Y':
    gridN = int(input('Enter the grid size N (2,4,6,8): '))

    if gridN not in [2, 4, 6, 8]:
        while gridN not in [2, 4, 6, 8]:
            gridN = int(input('Invalid grid size! Please re-enter the grid size N (2,4,6,8): '))

elif gridInput != 'N':
    while gridInput != 'Y' and gridInput != 'N':
        gridInput = input('Invalid answer! Please type Y or N: ')
        if gridInput == 'Y':
            gridN = int(input('Enter the grid size N (2,4,6,8): '))

            if gridN not in [2, 4, 6, 8]:
                while gridN not in [2, 4, 6, 8]:
                    gridN = int(input('Invalid grid size! Please re-enter the grid size N (2,4,6,8): '))

else:

    for instance in instances:
        filename = 'support_96/' + instance.strip()
        image = imread(filename)
        hist = utils.perChannelHistogram(image, interval)

        rHist.append(utils.l1Normalizer(hist[0]))
        gHist.append(utils.l1Normalizer(hist[1]))
        bHist.append(utils.l1Normalizer(hist[2]))

    # Query 1

    for j in range(1, 4):

        totalCorrect = 0

        for instance in instances:
            filename = 'query_' + str(j) + '/' + instance.strip()
            image = imread(filename)

            queryHist = utils.perChannelHistogram(image, interval)
            redChannelHist = utils.l1Normalizer(queryHist[0])
            greenChannelHist = utils.l1Normalizer(queryHist[1])
            blueChannelHist = utils.l1Normalizer(queryHist[2])

            lowestAt = -1
            minDivergence = 99999

            for i in range(len(instances)):
                divR = utils.JSDivergence(redChannelHist, rHist[i])
                divG = utils.JSDivergence(greenChannelHist, gHist[i])
                divB = utils.JSDivergence(blueChannelHist, bHist[i])

                div = (divR + divG + divB) / 3

                if div < minDivergence:
                    minDivergence = div
                    lowestAt = i

            if instance == instances[lowestAt]:
                totalCorrect += 1


        top1Acc = totalCorrect / len(instances)

        print(f'Top-1 accuracy of the query-{j} dataset is: {top1Acc}')
