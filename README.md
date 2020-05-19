# Inuktitut-Detect
Classifier between inuktitut and english words written with PyTorch


## Downloading the Dataset
The model is trained from [The Nunavut Hansard Inuktitut-English Parallel Corpus](https://www.inuktitutcomputing.ca/NunavutHansard/info.php)  
Find the corpus [here](https://www.inuktitutcomputing.ca/NunavutHansard/data/SentenceAligned.v2.txt.zip)  

## Training the Model
To train the model run the following:  
`python train.py corpus.txt param.json`  
  * `corpus.txt` is the location of the downloaded corpus file  
  * `param.json` is the location of the hyperparamater file  

#### Optional Arguments

`-v (default 1)` sets the verbosity  
  * `v>1` outputs the training progress every epoch  
  * `v=1` only outputs a final report  
  
  `-t` tells the program to retokienize the data. Since tokienization takes some time, by default it is saved to a created `/Data` folder in the form of `.npy` files. Adding this flag will reprocess the corpus.  

#### Paramater File  
  Paramater file is included in the repo. It includes the paramaters:  
  * `learning rate:0.005`  
  The learning rate our model uses during SGD  
  * `num epochs:100`    
The number of epochs in the training loop  
  * `num test:1000`  
The number of samples set aside for testing  

## Results

Using the hyperparameters included in params.json, a BCE of 0.12 and accuracy of 96%  was acheived on the test set.
![image](https://raw.githubusercontent.com/AlecCooper/Inuktitut-Detect/master/Results/results.png?token=ABFU22E6WORKYPGK64KXMPS6YQR36)
