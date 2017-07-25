#-*- encoding: utf-8 -*-

from PIL import Image
import numpy as np
import os
import pandas as pd
import pylab as pl
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.model_selection import train_test_split



STANDARD_SIZE = (500, 500)


def img_to_matrix(filename, verbose=False):
    img = Image.open(filename)
    if verbose:
        print ('changing size from %s to %s' % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    imgArray = np.asarray(img)
    return imgArray  # imgArray.shape = (167 x 300 x 3)


def flatten_image(img):
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def main():
    img_dir = 'images/'
    images = [img_dir + f for f in os.listdir(img_dir)]
    labels = ['salt' if 'salt' in f.split('/')[-1] else 'soy' for f in images]

    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)


    is_train = np.random.uniform(0, 1, len(data)) <= 0.8
    y = np.where(np.array(labels) == 'salt', 1, 0)

    train_x, train_y = data[is_train], y[is_train]


    # training a classifier
    number = 10
    pca = decomposition.PCA(n_components=number,whiten=True)
    train_x = pca.fit_transform(train_x,)

    svm = SVC(C=1.0)
    svm.fit(train_x, train_y)
    joblib.dump(svm, 'model.pkl')

    # evaluating the model
    test_x, test_y = data[is_train == False], y[is_train == False]
    test_x = pca.transform(test_x)
    print(str(number) + "の時：")
    print (pd.crosstab(test_y, svm.predict(test_x),
                      rownames=['Actual'], colnames=['Predicted']))

if __name__ == '__main__':
    main()