from granular_cond_entropy_FS import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import os


# determine proxy label for unlabeled data according to the prior class probability and the class ratio of labeled data
def proxy_label_fun(pr_u=0.2, n_all_examples=1000, alpha=0.1, beta=0.5, gamma=500, epsilon=0.0002):
    # prior part
    scale_weight = pow(1 + epsilon, n_all_examples)
    if pr_u <= 0.5:
        pr_prior = min(pr_u * scale_weight, 0.5)
    else:
        pr_prior = 1 - min((1 - pr_u) * scale_weight, 0.5)

    # initial part
    pr_init = pow(beta, 1 + pow(np.e, -gamma * epsilon * alpha * n_all_examples))

    # determination of proxy label
    pr_all = pr_init * pr_prior
    if pr_all > 0.5:
        y_proxy = 0
    elif pr_all < 0.5:
        y_proxy = 1
    else:
        y_proxy = int(pr_u <= 0.5)
    print(pr_prior, pr_init, pr_all, y_proxy)

    return y_proxy


# test the performance of semi-supervised feature selection on the selected data
def semi_supervised_FS_proxy_labels():
    # data file for feature selection
    data_name = './data sets/cardiotocography-FHR pattern.data'

    # cross-validation for performance evaluation**********
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=3)
    # clf = LinearSVC(max_iter=10000)  # linear SVM

    label_rates = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    cls_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]

    data = np.loadtxt(data_name, dtype=int, delimiter=',')
    (n_examples, m_features) = data.shape
    print(data.shape)

    # **********01: pre-process data**********
    X = data[:, 0:m_features - 1]
    y = data[:, -1].copy()

    # multi-class to binary in 1-vs-all manner, maximum class to 1, other classes to 0
    count_class = np.bincount(y)
    n_class = sum(count_class > 0)
    max_cls = np.argmax(count_class)

    y[y != max_cls] = -1
    y[y == max_cls] = 1
    y[y != 1] = 0
    data[:, -1] = y

    cls_y1_prb = np.mean(y)  # the prior probability of class '1' in all data
    pos_indices = np.where(y == 1)[0]  # indices of all positive examples, tuple with [0] to array
    neg_indices = np.where(y == 0)[0]  # indices of all negative examples

    # ********** different label rates **********
    label_rate = label_rates[2]               # the rate for labeled data over all dataset, 10% for examples
    label_num = int(n_examples * label_rate)  # the number of initially labeled data

    label_init = np.array([0.0] * len(cls_ratios))
    label_semi = np.array([0.0] * len(cls_ratios))

    # ********** different class ratio **********
    for j in range(0, len(cls_ratios)):
        label_y1_ratio = cls_ratios[j]    # the positive class ratio for labeled data

        if label_num <= 1:
            label_num = 2

        label_y1_num = int(cls_y1_prb * label_y1_ratio * label_num)  # num. of labeled positive examples
        if label_y1_num == 0:
            label_y1_num = 1
        if label_y1_num >= label_num:
            label_y1_num = label_num - 1
        label_y0_num = label_num - label_y1_num                      # num. of labeled negative examples
        label_cls_ratio = label_y1_num * 1.0 / label_y0_num          # ratio of positive vs negative

        # ********** different random seed **********
        random_init = np.array([0.0] * 10)
        random_semi = np.array([0.0] * 10)

        for rnd in range(0, 10):
            # shuffle the indices for initial labeled data
            np.random.seed(rnd)
            np.random.shuffle(pos_indices)
            np.random.shuffle(neg_indices)

            y_semi = np.array([-1] * len(y))
            y_semi[pos_indices[:label_y1_num]] = 1
            y_semi[neg_indices[:label_y0_num]] = 0

            # **********03: generate proxy label**********
            print('Label rate: ', label_rate, ', class ratio: ', label_cls_ratio, ', random: ', rnd)
            print("03: generate proxy label......")
            proxy_label = proxy_label_fun(cls_y1_prb, n_examples, label_rate, label_cls_ratio, 500, 0.0002)
            y_proxy = y_semi.copy()
            y_proxy[np.where(y_semi == -1)] = proxy_label

            # ******************************************************************
            # **********4: semi-supervised feature selection**********
            # ******************************************************************
            print("04: feature selection......")
            label_indices = np.where(y_semi != -1)[0]
            unlabel_indices = np.where(y_semi == -1)[0]

            idx_init, mi_init = grd_ce_fs_accel(X[label_indices], y_semi[label_indices])
            print("Init feature subset: ", len(idx_init), idx_init, "\n")

            idx_semi, mi_semi = grd_ce_fs_accel(X, y_proxy)
            print("Semi feature subset: ", len(idx_semi), idx_semi, "\n")

            # **********5: cross-validation for performance evaluation**********
            # ***********5-1: feature subset of initial labeled data************
            # ******************************************************************
            # obtain the dataset on the selected features
            num_fea = len(idx_init)
            features = X[:, idx_init]
            print(features.shape)

            init_acc = cross_val_score(clf, features, y, cv=kf)
            print(init_acc)
            print("Performance with init FS: %0.4f (+/- %0.4f)\n" % (init_acc.mean(), init_acc.std() * 2))

            # ***********5-2: feature subset of semi-supervised data************
            # ******************************************************************
            # obtain the dataset on the selected features
            features = X[:, idx_semi]
            print(features.shape)

            semi_acc = cross_val_score(clf, features, y, cv=kf)
            print(semi_acc)
            print("Performance with semi FS: %0.4f (+/- %0.4f)\n" % (semi_acc.mean(), semi_acc.std() * 2))

            random_init[rnd] = np.mean(init_acc)
            random_semi[rnd] = np.mean(semi_acc)

        label_init[j] = np.mean(random_init)
        label_semi[j] = np.mean(random_semi)
        print("init FS with different ratios: %0.4f (+/- %0.4f)\n" % (random_init.mean(), random_init.std() * 2))
        print("semi FS with different ratios: %0.4f (+/- %0.4f)\n" % (random_semi.mean(), random_semi.std() * 2))

    print("init FS with different label rates: %0.4f (+/- %0.4f)\n" % (label_init.mean(), label_init.std() * 2))
    print("semi FS with different label rates: %0.4f (+/- %0.4f)\n" % (label_semi.mean(), label_semi.std() * 2))

    print('Finished!\n')


if __name__ == '__main__':
    # test granular conditional entropy-based feature selection with proxy labels
    semi_supervised_FS_proxy_labels()
