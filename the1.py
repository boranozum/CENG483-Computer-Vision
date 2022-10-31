from matplotlib.image import imread
import numpy as np

support_image = imread('support_96/Acadian_Flycatcher_0016_887710060.jpg')
query_image1 = imread('query_1/Acadian_Flycatcher_0016_887710060.jpg')
query_image2 = imread('query_1/American_Redstart_0004_534938556.jpg')


def perChannelHistogram(image, interval):
    imageHeight = image.shape[0]

    result = np.zeros([3, interval], int)

    for i in range(imageHeight):

        for j in range(interval):
            result[0, j] += ((image[i][:, 0] >= interval * j) & (image[i][:, 0] < interval * (j + 1))).sum()
            result[1, j] += ((image[i][:, 1] >= interval * j) & (image[i][:, 1] < interval * (j + 1))).sum()
            result[2, j] += ((image[i][:, 2] >= interval * j) & (image[i][:, 2] < interval * (j + 1))).sum()

    return result


def l1Normalizer(hist):
    norm = np.linalg.norm(hist)
    return hist / norm


def KLDivergence(query, search):
    sumArray = query * np.log(query / search)
    return sumArray.sum()


def JSDivergence(query, search):
    lhs = 0.5 * KLDivergence(query, (query + search) / 2)
    rhs = 0.5 * KLDivergence(search, (query + search) / 2)

    return lhs+rhs

support_image_hist = perChannelHistogram(support_image, 8)
query_image1_hist = perChannelHistogram(query_image1, 8)
query_image2_hist = perChannelHistogram(query_image2, 8)

s_r = l1Normalizer(support_image_hist[0])
s_g = l1Normalizer(support_image_hist[1])
s_b = l1Normalizer(support_image_hist[2])

q1_r = l1Normalizer(query_image1_hist[0])
q1_g = l1Normalizer(query_image1_hist[1])
q1_b = l1Normalizer(query_image1_hist[2])

q2_r = l1Normalizer(query_image2_hist[0])
q2_g = l1Normalizer(query_image2_hist[1])
q2_b = l1Normalizer(query_image2_hist[2])

s_q1 = (JSDivergence(s_r, q1_r) + JSDivergence(s_g, q1_g) + JSDivergence(s_b, q1_b))
s_q2 = (JSDivergence(s_r, q2_r) + JSDivergence(s_g, q2_g) + JSDivergence(s_b, q2_b))

print(s_q2)

print(s_q1)