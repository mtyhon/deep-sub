# Training the Mixture Density Network

![alt text](https://github.com/mtyhon/deep-sub/raw/master/sample/MDN_Schematic.png "Mixture Density Network")


The script in this folder trains the network on a grid of models, which is provided separately [here](https://drive.google.com/open?id=1PyNfvBzy0hicOLWijxeMDxOVBDdCnroI), which currently uses MESA r12115. 



Libraries additionally required:
---

* tqdm


Running the script
===


Run run_training.py. The script accepts the following arguments:


* '--input_filename': Path to grid of models
* '--hidden_size': Number of neurons in intermediate network layers. Default is 512
* '--kernel_size': Receptive field size of convolutional neural network. Default is 5
* '--num_gaussians': Number of Gaussians used to estimate the output distribution. Default is 16
* '--batch_size': Size of a batch of models that is loaded when training the network. Default is 32
* '--init_lr': Initial learning rate of Adam optimizer. Default is 0.001
* '--num_epochs': Number of training iterations. Default is 500
* '--save_best': Boolean flag telling the script whether or not to save the best network during training. This is evaluated using a hold-out validation set. Default is True.




The default parameters are used for training the network presented in the paper.



Running training, an example command:
---


python run_training.py --input_filename 'grid_file_here'
