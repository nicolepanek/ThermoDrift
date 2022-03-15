# ThermoDrift™
<p align="center">
  <img src="images/thermodrift_logo.png" />
</p>

Hi! Welcome to **ThermoDrift**, a user friendly tool to classify protein sequences as Thermophilic, Mesophilic, or Psychrophilic. This tool can be used for prediction, but is also an open ended tool for people to play with and adapt for the tasks they need. Look at the figure below to see what temperatures these organisms live at.

<p align="center">
  <img src="images/figure_1.jpg" />
</p>

### Why Thermodrift: 
Thermodrift was created as an open source project that enables automated protein classification for thermophilic, mesophilic, or psychrophilic phenotypes to address the lack of any computational classifiers for thermostable protein prediction that are wisely accessible and cater to a scientific user base with little machine learning experience but a lot of enthusiasm for protein characterization. 


### Using the GUI:
GUI Explanation here

### Data processing from uniprot:
How data was manipulated from sequence to one hot encoded sequence


### Data processing for future model training:
Processed pytorch tensors of one-hot encoded protein sequences and their respective classification (thermophilic, mesophilic, psychrophilic) make up the X and y input tensors. 
X is a tensor of shape [2000, 500, 20]. This represents 2000 examples, where each example is a sequence of length 500 AA. Each AA in the sequence is one-hot encoded across the z-dimension. 
y is a tensor of shape [2000, 1]. This represents 2000 examples, where each example contains a final classification.

Functions to prepare data for training in CNN: 
split_data(X, y) takes inputs of X and y tensors. The tensors are split 80/20 into training and testing sets, respectively. 
The training tensor dataset ("trainset") is formed by concatenating the X_train and y_train tensors into a pytorch TensorDataset.
The testing tensor dataset ("testset") is formed by concatenating the X_test and y_test tensors into a pytorch TensorDataset.
The function returns trainset and testset.

make_dataloader(trainset, testset, batchsize) takes inputs of TensorDatasets trainset and testset, as well as an integer batchsize (number of examples to feed into CNN at a time). 
The training set data is shuffled after each epoch in order to prevent bias during training. The test set is not shuffled. 
The TensorDatasets are converted into pyrotch Dataloaders using torch.utils.data.DataLoader.
The function returns the dataloaders train_loader and test_loader that can be fed into the CNN for training. 

### Running training:
How to launch a training session
