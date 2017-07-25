#-*- encoding: utf-8 -*-

from PIL import Image
import numpy as np
import os
import pandas as pd
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.svm import LinearSVC


STANDARD_SIZE = (300, 167)


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

    # plot in 2 dimensions
    pca = PCA(n_components=2)
    X = pca.fit_transform(data,)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1],
                       "label": np.where(y == 1, 'salt', 'soy')})
    colors = ['red', 'yellow']
    for label, color in zip(df['label'].unique(), colors):
        mask = df['label'] == label
        pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)

    pl.legend()
    pl.savefig('pca_feature.png')

    # training a classifier
    number = 10
    pca = PCA(n_components=number)
    train_x = pca.fit_transform(train_x,)

    svm = LinearSVC(C=1.0)
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