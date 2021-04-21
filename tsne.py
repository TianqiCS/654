import collections
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from time import time

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from models.dp_wgan_mse import DP_WGAN


def worker(params, x_train, y_train, x_test, y_test, input_dim, z_dim, class_ratios):
    model = DP_WGAN(input_dim, z_dim, target_epsilon=10000, target_delta=1e-3, conditional=True)

    model.train(x_train, y_train, params, private=True)

    syn_data = model.generate(x_train.shape[0], class_ratios=class_ratios)
    return test(syn_data, x_test, y_test)


def test(syn_data, x_test, y_test):
    x_syn, y_syn = syn_data[:, :-1], syn_data[:, -1]
    sample_features = x_syn
    sample_class = y_syn

    scr = StandardScaler()
    sample_features = scr.fit_transform(sample_features)
    print(sample_features.shape, sample_class.shape)
    model = TSNE(n_components=2, random_state=0, perplexity=30)

    t0 = time()
    embedded_data = model.fit_transform(sample_features)
    print("TSNE done in %0.3fs." % (time() - t0))

    final_data = np.concatenate((embedded_data, np.expand_dims(sample_class, axis=1)), axis=1)
    print(final_data.shape)
    newdf = pd.DataFrame(data=final_data, columns=["Dim1", "Dim2", "Class"])

    sns.FacetGrid(newdf, hue="Class", size=6).map(plt.scatter, "Dim1", "Dim2").add_legend()
    plt.title("Perplexity=30 with normalization")
    plt.show()


def main():
    root_path = "data/"
    df = pd.read_csv(root_path + "creditcard.csv")  #total 284,807 transactions only 492 frauds
    train_df, val_df = train_test_split(df, test_size=0.05)

    training_positive_samples = train_df.loc[train_df["Class"] == 1]
    training_negative_samples = train_df.loc[train_df["Class"] == 0].sample(n=500)
    train_df = pd.concat([training_positive_samples,training_negative_samples])

    val_positive_samples = val_df.loc[val_df["Class"] == 1]
    val_negative_samples = val_df.loc[val_df["Class"] == 0].sample(n=25)
    val_df = pd.concat([val_positive_samples,val_negative_samples])

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

    params = Hyperparams(batch_size=64, micro_batch_size=8,
                         clamp_lower=-0.01, clamp_upper=0.01,
                         clip_coeff=0.1, sigma=0.8, class_ratios=class_ratios, lr=1e-4,
                         num_epochs=10)
    worker(params, x_train, y_train, x_test, y_test, input_dim, z_dim, class_ratios)

Hyperparams = collections.namedtuple(
    'Hyperparams',
    'batch_size micro_batch_size clamp_lower clamp_upper clip_coeff sigma class_ratios lr num_epochs')
if __name__ == "__main__":
    main()
