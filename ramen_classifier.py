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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import ImageOps

np.random.seed(123)


STANDARD_SIZE = (128, 128)
CROP_RATIO = (0.7,0.7)


def img_to_matrix(filename, verbose=False):
    img = Image.open(filename)
    crop_size = (int(img.size[0] * CROP_RATIO[0]),int(img.size[1] * CROP_RATIO[1]))
    img = ImageOps.fit(img,crop_size)
    if verbose:
        print ('changing size from %s to %s' % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    imgArray = np.asarray(img)
    return imgArray  # imgArray.shape = (167 x 300 x 3)


def flatten_image(img):
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def main(pca_dim):
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
    y_all = np.where(np.array(labels) == 'salt', 1, 0)

    X = data[is_train]
    y = y_all[is_train]
    X_test = data[is_train == False]
    y_test = y_all[is_train == False]

    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits)
    accuracies = []
    test_probas = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print("[Fold {}/{}]".format(i, n_splits))
        train_x = X[train_idx]
        train_y = y[train_idx]
        val_x = X[val_idx]
        val_y = y[val_idx]

        # training a classifier
        pca = decomposition.PCA(n_components=pca_dim,whiten=True)
        train_x = pca.fit_transform(train_x,)

        svm = SVC(C=1.0, probability=True, verbose=True)
        svm.fit(train_x, train_y)
        model_path = 'model_{}.pkl'.format(i)
        joblib.dump(svm, model_path)

        # evaluating the model
        # test_x, test_y = data[is_train == False], y[is_train == False]
        val_x = pca.transform(val_x)
        print(str(pca_dim) + "の時：")
        val_pred = svm.predict(val_x)
        print (pd.crosstab(val_y, svm.predict(val_x),
                          rownames=['Actual'], colnames=['Predicted']))
        accuracies.append(accuracy_score(val_y, val_pred))

        reduced_X_test = pca.transform(X_test)
        test_probas.append(svm.predict_proba(reduced_X_test))

    accuracies = np.asarray(accuracies)
    print("Cross validation score: accuracy={0:.3f}+/-{1:.3f}".format(accuracies.mean(), accuracies.std()))

    print("[ Test ]")
    test_proba = np.asarray(test_probas).mean(axis=0)
    test_pred = np.argmax(test_proba, axis=1)
    accuracy = accuracy_score(y_test, test_pred)
    print("accuracy={0:.3f}".format(accuracy))
    return { "accuracy": accuracy }

if __name__ == '__main__':
    test_accuracies = []
    cross_val_accuracies = []
    pca_dim_space = list(range(2, 5+1))
    for d in pca_dim_space:
        print("\n\n")
        res = main(pca_dim=d)
        accuracy = res['accuracy']
        test_accuracies.append(accuracy)
    import matplotlib; matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure()
    plt.figure("Accuracy in each PCA dimension")
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('PCA dimension')
    plt.plot(pca_dim_space, test_accuracies)
    plt.plot(pca_dim_space,)
    plt.show()