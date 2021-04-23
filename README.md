# Exploring  Better Trade-off in Private Data Generation between Differential Privacy Guarantee and Generation Utility

This project is based on codes from [BorealisAI](https://github.com/BorealisAI/private-data-generation "private-data-generation") with major updates in model, dataset, loss function and additional visualization feature.

Our codes are divided in several sections, please check the following instructions on how to recreate our results.




## Dataset description and Models:

### Credit card dataset
The datasets contains transactions made by credit cards in September 2013 by european cardholders. 
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. In the experiment, we sample the dataset to get balanced data for model training. To get access to the dataset, download at https://www.kaggle.com/mlg-ulb/creditcardfraud

### MNIST dataset

We leverage the MNIST dataset directly from Pytorch, i.e., torchvision.dataset. For the binary classificaiton task, we select images in class 0 and 1 and the number of each class is balanced. The dimension of each images is 28*28 and we transform them into N*784 tabular form into  .csv files, but transform them back to 28*28 while the training of CNN-based GANs. The resulting dataset contains 12,665 training data and 2,115 test data. In particular, go to the subfolder "DPWGAN-PATEGAN-MNIST/preprocessing". Run "python process_MNIST_pytorch.py". This code downloads and then splits the MNIST dataset into trainset and testset automatically.

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

**DP-WGAN on MNIST dataset** : Go to the subfolder ""DPWGAN-PATEGAN-MNIST/", run the following code：

···
python evaluate.py --epoch 20 --target-variable='y' --train-data-path=./data/MNIST_CNN_train.csv --test-data-path=./data/MNIST_CNN_test.csv --normalize-data real-data --enable-privacy --sigma=1.0
···

Note that we fix the training epoch and noise size 'sigma', and then we record the resulting epsilon in DP and evaluate the classfication performance on the generated data.

**DP-WGAN on MNIST dataset** : Go to the subfolder ""DPWGAN-PATEGAN-MNIST/", run the following code：

···
python evaluate.py --epoch 2000 --target-variable='y' --train-data-path=./data/MNIST_CNN_train.csv --test-data-path=./data/MNIST_CNN_test.csv --normalize-data pate-gan --enable-privacy --num-teachers 5 --lap-scale 1e-4
···

Note that we fix the training epoch and noise size 'lap-scale', and then we record the resulting epsilon in DP and evaluate the classfication performance on the generated data. You can adjust hyparameters, e.g., noise size and epoch, to satisfy your need.












