import collections
import multiprocessing as mp

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score

from tabulate import tabulate

from models.dp_wgan_mse import DP_WGAN


def worker(params, x_train, y_train, x_test, y_test, input_dim, z_dim, class_ratios):
    model = DP_WGAN(input_dim, z_dim, target_epsilon=10000, target_delta=1e-3, conditional=True)

    model.train(x_train, y_train, params, private=True)

    syn_data = model.generate(x_train.shape[0], class_ratios=class_ratios)
    return test(syn_data, x_test, y_test)


def test(syn_data, x_test, y_test):
    x_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]

    # test the results on learners
    learners = []
    names = ['BernoulliNB', 'Random Forest', 'AdaBoostClassifier', 'ExtraTreesClassifier',
             'GradientBoostingClassifier']
    learners.append((BernoulliNB()))
    learners.append((RandomForestClassifier()))
    learners.append((AdaBoostClassifier()))
    learners.append((ExtraTreesClassifier()))
    learners.append((GradientBoostingClassifier()))
    result = {}
    for i in range(0, len(learners)):
        learners[i].fit(x_syn, y_syn)
        pred_probs = learners[i].predict_proba(x_test)
        auc_score = roc_auc_score(y_test, pred_probs[:, 1])
        result[names[i]] = auc_score
    return result


def main():
    # load dataset
    root_path = "../data/"
    df = pd.read_csv(root_path + "creditcard.csv")  #total 284,807 transactions only 492 frauds
    train_df, val_df = train_test_split(df, test_size=0.05)

    # balance data
    training_positive_samples = train_df.loc[train_df["Class"] == 1]
    training_negative_samples = train_df.loc[train_df["Class"] == 0].sample(n=500)
    train_df = pd.concat([training_positive_samples,training_negative_samples])

    val_positive_samples = val_df.loc[val_df["Class"] == 1]
    val_negative_samples = val_df.loc[val_df["Class"] == 0].sample(n=25)
    val_df = pd.concat([val_positive_samples,val_negative_samples])

    # preprocessing
    class_ratios = train_df["Class"].sort_values().groupby(train_df["Class"]).size().values/train_df.shape[0]
    print("class ratio", class_ratios)
    x_train = np.nan_to_num(train_df.drop(["Class"], axis=1).values)
    y_train = np.nan_to_num(train_df["Class"].values).T
    x_test = np.nan_to_num(val_df.drop(["Class"], axis=1).values)
    y_test = np.nan_to_num(val_df["Class"].values).T

    input_dim = x_train.shape[1]
    z_dim = int(input_dim / 4 + 1) if input_dim % 4 == 0 else int(input_dim / 4)

    print(x_train.shape, y_train.shape)
    print(x_test.shape,y_test.shape)

    # setting hyper params
    params = Hyperparams(batch_size=64, micro_batch_size=8,
                         clamp_lower=-0.01, clamp_upper=0.01,
                         clip_coeff=0.1, sigma=0.8, class_ratios=class_ratios, lr=1e-4,
                         num_epochs=10)

    # multi processing and take the avg of all runs
    pool = mp.Pool(processes=6)
    tasks = [pool.apply_async(worker, args=(params, x_train, y_train, x_test, y_test, input_dim, z_dim, class_ratios)) for i in range(6)]
    results = [np.array(list(p.get().values())) for p in tasks]
    avg = np.mean(results, axis=0)

    table = [['BernoulliNB', 'Random Forest', 'AdaBoostClassifier', 'ExtraTreesClassifier',
             'GradientBoostingClassifier']]
    table += results
    table += [avg]
    print("Showing results of epochs = {}, sigma = {}".format(params.num_epochs, params.sigma))
    print(tabulate(table, headers='firstrow', floatfmt=".2f"))

Hyperparams = collections.namedtuple(
    'Hyperparams',
    'batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs')
if __name__ == "__main__":
    main()

