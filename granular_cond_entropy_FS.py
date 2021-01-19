import numpy as np
import math


# merge a matrix with discrete variables in column to a new discrete variable
def merge_matrix(mat_X):
    assert mat_X.size != 0

    new_x, idx = np.unique(mat_X, axis=0, return_index=True)
    n_es = new_x.shape[0]
    n_fs = new_x.shape[1]
    new_vec = np.array(
        [np.where((new_x == x.repeat(n_es).reshape(-1, n_es).T).sum(axis=1) == n_fs)[0] for x in mat_X]).flatten()

    return new_vec


# merge two discrete variables to a new discrete variable
def merge_two_variables(a, b):
    assert a.shape == b.shape

    c = np.vstack((a, b)).T   # row vectors stack, convert to column vectors
    unique_c, idx = np.unique(c, axis=0, return_index=True)
    new_vec = np.array([np.where((x[0] == unique_c[:, 0]) & (x[1] == unique_c[:, 1])) for x in c]).flatten()

    return new_vec


# entropy of a discrete random variable
# ent = - sum{pi*log(pi, base=2)}
def entropy(vec, base=2):
    assert vec.ndim == 1   # vector only

    vec_len = len(vec)
    unique_v = np.unique(vec)

    ent = 0
    for i in range(0, len(unique_v)):
        pi = float(sum(vec == unique_v[i])) / vec_len
        if pi >= 1.:
            ent = 0
            break
        else:
            ent = ent - pi * math.log(pi, base)

    return ent


# granular conditional entropy between two discrete random variables
# GH(D|C) = -sum_{Ci}[P(Ci)^2 * sum_{Dj}[P(Dj|Ci)logP(Dj|Ci)]]
def gra_cond_e(x, y, rows_n, base=2):
    assert x.shape == y.shape

    # compute the equivalence classes of x
    unique_x = np.unique(x)

    gce = 0
    all_indices = []
    for i in range(0, len(unique_x)):
        # print(unique_x[i])

        # get the indices of each condition equivalence class
        indices = np.where(x == unique_x[i])[0]
        dec_eqv_class = y[indices]
        H_Ci = entropy(dec_eqv_class)
        gce = gce + pow(float(len(dec_eqv_class)) / float(rows_n), 2) * H_Ci

        # for accelerator
        if H_Ci <= 0:
            all_indices = np.hstack((all_indices, indices))

    # truncate the float number to avoid precision error
    gce = round(gce, 10)

    return gce, all_indices


# feature selection based on granular conditional entropy with accelerator
# note: the result may be slightly different if you use other way to computing entropy since the precision in float
def grd_ce_fs_accel(X, y):
    n_rows, n_cols = X.shape
    print(X.shape)

    # all features to one vector
    vector_all = merge_matrix(X)

    # calculate granular conditional entropy between all features and decision class
    all_gc, _ = gra_cond_e(vector_all, y, n_rows)
    print('Overall entropy:', all_gc)

    # calculate granular conditional entropy between each feature and decision class
    fea_gc = np.zeros(n_cols)   # relevance
    fea_rd = np.zeros(n_cols)   # redundancy
    Feas = np.ones(n_cols)      # flag for candidate features

    for i in range(n_cols):
        f = X[:, i]
        fea_gc[i], _ = gra_cond_e(f, y, n_rows)

    cur_gc, idx, err_thres = 0, 0, 1E-12
    F = []         # index of selected features
    feas_gc = []   # information entropy between iteratively selected features and decision class
    f_select = []  # all selected features in one vector manner

    del_list = list()  # used to remove examples whose entropy is zero
    while True:
        if len(F) == 0:
            # first time selection: select the feature whose entropy is the smallest
            idx = np.argmin(fea_gc)
            F.append(idx)
            Feas[idx] = 0
            cur_gc = fea_gc[idx]
            feas_gc.append(cur_gc)
            f_select = X[:, idx]

        if abs(all_gc - cur_gc) < err_thres:
            break

        del_list.clear()
        for i in range(n_cols):
            if Feas[i]:
                f = X[:, i]
                new_f = merge_two_variables(f_select, f)
                fea_gc[i], del_indices = gra_cond_e(new_f, y, n_rows)
                fea_rd[i], _ = gra_cond_e(f_select, f, n_rows)
                del_list.append(del_indices)
            else:
                fea_gc[i] = 88      # already selected features
                fea_rd[i] = 99      # maximum redundancy
                del_list.append([])

        # select the corresponding feature index with the largest relevance, minimum redundancy
        cur_gc = min(fea_gc)
        indices = np.where((fea_gc >= cur_gc - err_thres) & (fea_gc <= cur_gc + err_thres))[0]  # avoid precision error
        idx = indices[np.argmax(fea_rd[indices])]   # rank the feature by redundancy further
        F.append(idx)
        Feas[idx] = 0
        feas_gc.append(cur_gc)
        f_select = merge_two_variables(f_select, X[:, idx])  # covert all selected feature vectors to one vector

        # *********** accelerator **********
        # remove examples
        del_indices = del_list.pop(idx)
        if len(del_indices) != 0:
            X = np.delete(X, del_indices, axis=0)
            y = np.delete(y, del_indices)
            f_select = np.delete(f_select, del_indices)

        # remove features
        for i in range(n_cols):
            if fea_rd[i] == 0:
                Feas[i] = 0

    return F, feas_gc


# feature selection based on granular conditional entropy
def gra_ce_fs(X, y):
    n_rows, n_cols = X.shape
    print(X.shape)

    # all features to one vector
    vector_all = merge_matrix(X)

    # calculate granular conditional entropy between all features and target class
    all_ce, _ = gra_cond_e(vector_all, y, n_rows)

    # calculate gce between each feature and target class
    fea_ce = np.zeros(n_cols)
    for i in range(n_cols):
        f = X[:, i]
        fea_ce[i], _ = gra_cond_e(f, y, n_rows)

    cur_ce, temp_ce, idx, err_thres = 0, 0, 0, 1E-12
    F = []         # index of selected features
    feas_ce = []   # gce between iteratively selected features and decision class: relevance
    f_select = []  # data with all selected features in one vector

    while True:
        if len(F) == 0:
            # first time selection: select the feature whose gce is the largest
            idx = np.argmin(fea_ce)
            F.append(idx)
            cur_ce = fea_ce[idx]
            feas_ce.append(cur_ce)
            f_select = X[:, idx]
            # print(fea_ce)

        if abs(all_ce - cur_ce) < err_thres:
            break

        cur_ce = 1E8
        for i in range(n_cols):
            if i not in F:
                f = X[:, i]
                new_f = merge_two_variables(f_select, f)
                temp_ce, _ = gra_cond_e(new_f, y, n_rows)
                # print(i, temp_ce)

                # select the largest gce and the corresponding feature index
                if temp_ce < cur_ce and abs(cur_ce - temp_ce) > err_thres:  # avoid error in float precision.
                    cur_ce = temp_ce
                    idx = i
        F.append(idx)
        feas_ce.append(cur_ce)
        f_select = merge_two_variables(f_select, X[:, idx])  # covert all selected feature vectors to one vector
        # print('Selected: ', idx, cur_ce)

    return F, feas_ce