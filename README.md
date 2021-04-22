# Exploring  Better Trade-off in Private Data Generation \\ between Differential Privacy Guarantee and Generation Utility

This project is based on codes from [BorealisAI](https://github.com/BorealisAI/private-data-generation "private-data-generation") with major updates in model, dataset, loss function and additional visualization feature.

Our codes are divided in several sections, please check the following instructions on how to recreate our results.




## Dataset description :

## Models : 

### Credit card dataset
**PATE-GAN** : PATE-GAN : Generating Synthetic Data with Differential Privacy Guarantees. ICLR 2019

**DP-WGAN** : Implementation of private Wasserstein GAN using noisy gradient descent moments accountant. 

### MNIST dataset

## How to:

**PATE-GAN on Credit card dataset** : Go to the subfolder "PATE-GAN-For-Credit-dataset" and run the note book.
Notice, you need to change the "read file directory" from the first part of notebook with your own settings.

For example: 
For the following 2 lines of codes, 

root_path = "/content/drive/My Drive/654/"

df = pd.read_csv(root_path + "creditcard.csv")

you can change the root_path of your own version, and make sure you put the file under the path you changed to.















