#-*- encoding: utf-8 -*-

from PIL import Image
import numpy as np
import os
import pandas as pd
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from PIL import ImageOps
import time

np.random.seed(123)


STANDARD_SIZE = (128, 128)
CROP_RATIO = (0.3,0.3)


def img_to_matrix(filename, verbose=False):
    img = Image.open(filename)
    crop_size = (int(img.size[0] * CROP_RATIO[0]),int(img.size[1] * CROP_RATIO[1]))
    img = ImageOps.fit(img,crop_size)
    if verbose:
        print ('changing size from %s to %s' % (str(img.size), str(STANDARD_SIZE)))
    img = img.resize(STANDARD_SIZE)
    imgArray = np.asarray(img)
    return imgArray  # imgArray.shape = (128 x 128 x 3)


def flatten_image(img):
    s = img.shape[0] * img.shape[1] * img.shape[2]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def main(X,y,X_test,y_test,pca_dim):
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits)
    accuracies = []
    test_probas = []

    print(str(pca_dim) + "次元圧縮：")
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print("[Fold {}/{}]".format(i+1, n_splits))
        train_x = X[train_idx]
        train_y = y[train_idx]
        val_x = X[val_idx]
        val_y = y[val_idx]

        # training a classifier
        pca = decomposition.PCA(n_components=pca_dim,whiten=True)
        train_x = pca.fit_transform(train_x,)

        svm = SVC(C=1.0, probability=True)
        svm.fit(train_x, train_y)
        model_path = 'model_{}.pkl'.format(i)
        joblib.dump(svm, model_path)

        # evaluating the model
        # test_x, test_y = data[is_train == False], y[is_train == False]
        val_x = pca.transform(val_x)
        val_pred = svm.predict(val_x)
        print (pd.crosstab(val_y, svm.predict(val_x),
                          rownames=['Actual'], colnames=['Predicted']))
        accuracies.append(accuracy_score(val_y, val_pred))

        reduced_X_test = pca.transform(X_test)
        test_probas.append(svm.predict_proba(reduced_X_test))

    accuracies = np.asarray(accuracies)
    mean_accuracies = accuracies.mean()
    print("Cross validation accuracy = {0:.3f}+/-{1:.3f}".format(mean_accuracies, accuracies.std()))


    test_proba = np.asarray(test_probas).mean(axis=0)
    test_pred = np.argmax(test_proba, axis=1)
    test_accuracy = accuracy_score(y_test, test_pred)
    print("Test accuracy = {0:.3f}".format(test_accuracy))

    test_cross_loss = abs(test_accuracy - mean_accuracies)
    print("Absolute(Test - Cross) loss = {0:.3f}".format(test_cross_loss))

    return { "test_accuracy": test_accuracy,"cross_val_accuracy":mean_accuracies,"test_cross_loss":test_cross_loss}

if __name__ == '__main__':
    #諸々の前処理
    img_dir = 'images/'
    images = [img_dir + f for f in os.listdir(img_dir)]
    labels = ['salt' if 'salt' in f.split('/')[-1] else 'soy' for f in images]

    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)

    # set the ratio about which one is train or test data
    is_train = np.random.uniform(0, 1, len(data)) <= 0.8
    y_all = np.where(np.array(labels) == 'salt', 1, 0)

    # for train data
    X = data[is_train]
    y = y_all[is_train]
    # for test data
    X_test = data[is_train == False]
    y_test = y_all[is_train == False]



    test_accuracies = []
    cross_val_accuracies = []
    test_cross_losses = []

    pca_dim_space = list(range(2,64+1))
    start = time.time()
    for d in pca_dim_space:
        print("\n\n")
        res = main(X,y,X_test,y_test,pca_dim=d)

        accuracy = res['test_accuracy']
        test_accuracies.append(accuracy)
        cross_val_accuracy = res['cross_val_accuracy']
        cross_val_accuracies.append(cross_val_accuracy)
        test_cross_loss = res['test_cross_loss']
        test_cross_losses.append(test_cross_loss)

    elapsed_time = time.time() - start
    print("")
    print("========== finished ==========")
    print("")
    print("elapsed_time = {0}".format(elapsed_time) + "[sec]")

    #show graph
    import matplotlib; matplotlib.use('MacOSX')
    import matplotlib.pyplot as plt
    fig = plt.figure("Test and Cross-Validation accuracy in each PCA dimension")
    plt.ylim(0.55, 0.8)
    plt.ylabel('Accuracy')
    plt.xlabel('PCA dimension')
    #plt.plot(pca_dim_space, test_accuracies)
    #plt.plot(pca_dim_space,cross_val_accuracies)


    print("Max Test accuracy = " + str(max(test_accuracies)))
    print("Min (Test - Cross) loss = " + str(min(test_cross_losses)))
    line1, = plt.plot(pca_dim_space, test_accuracies, label="Test accuracy", linestyle='--')
    line2, = plt.plot(pca_dim_space, cross_val_accuracies, label="Cross-Validation accuracy", linewidth=4)


    # Create a legend for the first line.
    first_legend = plt.legend(handles=[line1], loc=1)

    # Add the legend manually to the current Axes.
    ax = plt.gca().add_artist(first_legend)

    # Create another legend for the second line.
    plt.legend(handles=[line2], loc=4)

    plt.show()