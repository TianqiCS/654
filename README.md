# Exploring  Better Trade-off in Private Data Generation \\ between Differential Privacy Guarantee and Generation Utility

This project is based on codes from [BorealisAI](https://github.com/BorealisAI/private-data-generation "private-data-generation") with major updates in model, dataset, loss function and additional visualization feature.

Our codes are divided in several sections, please check the following instructions on how to recreate our results.




## Dataset description and Models:

### Credit card dataset
The datasets contains transactions made by credit cards in September 2013 by european cardholders. 
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. In the experiment, we sample the dataset to get balanced data for model training. To get access to the dataset, download at https://www.kaggle.com/mlg-ulb/creditcardfraud

### MNIST dataset

**PATE-GAN** : PATE-GAN : Generating Synthetic Data with Differential Privacy Guarantees. ICLR 2019

**DP-WGAN** : Implementation of private Wasserstein GAN using noisy gradient descent moments accountant. 

## How to:

**DP-WGAN on Credit card dataset** : In the subfolder "DP-WGAN-For-Credit-dataset" and run credit.py. You can adjust parameters and multiprocessing workflow in the file. 
To generate TSNE visualization, you can run tsne.py and adjust the parameters there.

**PATE-GAN on Credit card dataset** : Go to the subfolder "PATE-GAN-For-Credit-dataset" and run the note book.
Notice, you need to change the "read file directory" from the first part of notebook with your own settings.

For example: 
For the following 2 lines of codes, 

root_path = "/content/drive/My Drive/654/"

df = pd.read_csv(root_path + "creditcard.csv")

you can change the root_path of your own version, and make sure you put the file under the path you changed to.















