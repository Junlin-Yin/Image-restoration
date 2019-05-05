import numpy as np
from imageio import imread, imsave
import scipy.stats as stats
normpdf = stats.norm.pdf


def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def imwrite(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype == np.double:
        # img = np.array(img*255, dtype=np.uint8)
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imsave(filename, img)


orifolder = '../ori'
datafolder = '../data'
samples = {'1': 0.8, '2': 0.4, '3': 0.6}
for testName, noiseRatio in samples.items():
    Img = im2double(imread('{}/{}.png'.format(orifolder, testName)))
    Img[(Img == 0)] = 0.001
    # ############## generate corrupted image ###############

    if len(Img.shape) == 2:
        # only black-white images
        Img = Img[:, :, np.newaxis]
    rows, cols, channels = Img.shape

    # generate noiseMask and corrImg
    noiseMask = np.ones((rows, cols, channels))
    subNoiseNum = round(noiseRatio * cols)
    for k in range(channels):
        for i in range(rows):
            tmp = np.random.permutation(cols)
            noiseIdx = np.array(tmp[:subNoiseNum])
            noiseMask[i, noiseIdx, k] = 0
    corrImg = Img * noiseMask
    imwrite(corrImg, '{}/{}.png'.format(datafolder, testName))
