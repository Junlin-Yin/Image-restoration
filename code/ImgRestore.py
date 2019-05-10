import numpy as np
import numpy.linalg as lg
from imageio import imread, imsave
import scipy.stats as stats
# import matplotlib.pyplot as plt
# import h5py
normpdf = stats.norm.pdf


def im2double(im):
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def imwrite(im, filename):
    img = np.copy(im)
    img = img.squeeze()
    if img.dtype == np.double:
        img = img * np.iinfo(np.uint8).max
        img = img.astype(np.uint8)
    imsave(filename, img)


orifolder = '../ori'
datafolder = '../data'
resfolder = '../result'
restoreRatio = 0
samples = {'A': 0.8, 'B': 0.4, 'C': 0.6}
for testName, noiseRatio in samples.items():
    # Img = im2double(imread('{}/{}.png'.format(orifolder, testName)))
    corrImg = im2double(imread('{}/{}.png'.format(datafolder, testName)))
    if len(corrImg.shape) == 2:
        # only black-white images
        # Img = Img[:, :, np.newaxis]
        corrImg = corrImg[:, :, np.newaxis]
    rows, cols, channels = corrImg.shape
    noiseMask = np.array(corrImg != 0, dtype='double')

    # standardize the corrupted image
    minX = np.min(corrImg)
    maxX = np.max(corrImg)
    corrImg = (corrImg - minX)/(maxX-minX)  # corrImg: 0~1

    basisNum = 50   # number of basis functions
    sigma = 0.01    # standard deviation
    lmda = 0.001    # lambda value to regularize weights
    # mean value of each basis function
    Phi_mu = np.linspace(1, cols, basisNum)/cols
    # set the standard deviation to the same value for brevity
    Phi_sigma = sigma * np.ones(basisNum)
    X = np.linspace(0, 1, num=cols)
    Phi = [[(np.exp(- (X[n] - Phi_mu[m]) ** 2) / (2 * Phi_sigma[m] ** 2)) for m in range(basisNum)] for n in range(cols)]
    Phi = np.array(Phi)
    lmda_I = lmda * np.eye(basisNum, dtype='double')
    resImg = np.copy(corrImg)

    for ch in range(channels):
        # in each channels
        for r in range(rows):
            # in each row
            # prepare for the transformation coeficiency
            p_cols = np.nonzero(noiseMask[r, :, ch] != 0)[0]
            n_cols = np.nonzero(noiseMask[r, :, ch] == 0)[0]
            Phi_p = Phi[p_cols]
            Phi_n = Phi[n_cols]
            TransCoe = lg.pinv(Phi_p.transpose().dot(Phi_p) + lmda_I).dot(Phi_p.transpose())

            # restore the missing pixels
            t_p = resImg[r, :, ch][p_cols]
            weight_ML = TransCoe.dot(t_p)
            t_n = Phi_n.dot(weight_ML)
            for c in range(len(n_cols)):
                resImg[r, n_cols[c], ch] = t_n[c]

    # show the restored image
    # if resImg.shape[2] == 1:
    #     Img = Img.squeeze()
    #     corrImg = corrImg.squeeze()
    #     resImg = resImg.squeeze()
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(Img, cmap='gray')
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(corrImg, cmap='gray')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(resImg, cmap='gray')
    #     plt.show()
    # else:
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(Img)
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(corrImg)
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(resImg)
    #     plt.show()

    # prefix = '%s/%s_%.1f_%d'%(resfolder, testName, noiseRatio, basisNum)
    # fileName = prefix+'_result.h5'
    # h5f = h5py.File(fileName,'w')
    # h5f.create_dataset("basisNum", dtype='uint8', data=basisNum)
    # h5f.create_dataset("sigma", dtype='double', data=sigma)
    # h5f.create_dataset("Phi_mu", dtype='double', data=Phi_mu)
    # h5f.create_dataset("Phi_sigma", dtype='double', data=Phi_sigma)
    # h5f.create_dataset("resImg", dtype='double', data=resImg)
    # h5f.create_dataset("noiseMask", dtype='double', data=noiseMask)
    # h5f.close()
    # h5f = h5py.File(fileName,'r')
    # resImg = h5f.get('resImg').value
    # noiseMask = h5f.get('noiseMask').value
    # h5f.close()

    # compute error
    # im1 = Img.flatten()
    # im2 = corrImg.flatten()
    # im3 = resImg.flatten()
    # norm12 = lg.norm(im1-im2, 2)
    # norm13 = lg.norm(im1-im3, 2)
    # print((
    #     '{}({}):\n'
    #     'Distance between original and corrupted: {}\n'
    #     'Distance between original and reconstructed (regression): {}'
    #     'Restore ratio: {}\n'
    # ).format(testName, noiseRatio, norm12, norm13, norm13/norm12))
    # restoreRatio += norm13/norm12

    # store figure
    imwrite(resImg, '{}/{}.png'.format(resfolder, testName))
# print('Average Restore Ratio:', restoreRatio/3)
