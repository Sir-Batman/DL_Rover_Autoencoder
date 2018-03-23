AutoEncoder
===========

Scripts - CAE
----
`train_cae.py` - Trains the Convolutional Autoencoder. Select which encoder to use with the `model_file` parameter. The trained model parameters are saved in the folder of the model.

`encode_laser.py` - A sample file showing how to load the autoencoder and call it on different laser scan data. Now saves a sample input, encoding, and decoding to the corresponding model folder.

`LaserDataset.py` - LaserDataset class used to load laser data

`L1Penalty.py` - Used for experimenting with Sparse Autoencoders.

Scripts - Autoencoder
----
Because of some differences with how a plain autoencoder works, we created some equivalent scripts that work for the plain autoencoder

`train_ae.py` - Trains the linear autoencoder.

`encode_laser_ae.py` - `encode_laser.py` equivalent for linear autoencoder


Models
----
The `models` folder contains various iterations of our autoencoder. Read the README for each one to get a sense of what they are.

Each model in the model folder contains the Autoencoder Class in `laser_cae.py` and the model parameters in `conv_autoencoder.pth`.

It also contains some log files for how the encoder was trained.

`model0` - Our initial autoencoder that encodes to 10 vector. Still poor decode performance.

`model1` - First iteration. Now encodes to 8 vector. Still poor decode performance.

`model6` - Clone of first iteration, contains files/data used in our presentation.

`model7` - Second iteration. Removes Batch Norms and Max Pooling.

Samples
----

`samples/sample_poi_laser.csv` - Contains 10 POI laser scans used to sanity check the model (samples not used to train model)

`samples/sample_rov_laser.csv` - Contains 10 ROV laser scans used to sanity check the model (samples not used to train model)

`samples/train_poi_laser.csv` - Contains 2 POI laser scans used to sanity check the model (samples from 100,000 used to train model)

`samples/train_rov_laser.csv` - Contains 2 ROV laser scans used to sanity check the model (samples from 100,000 used to train model)


Spreadsheet
----
`layer_size_calc.ods` - spreadsheet that computes the output size of each layer given parameters.


Current Status
----
It seems that the autoencoder's performance does not work very well when given 100,000 samples to train on. This can be seen in `cae_training.log`, which shows that the loss while the autoencoder is trained does not decrease. Additionally, looking at `test.log` and `train.log`, the decoded laser scan does not seem very accurate. 

More development of the network is necessary.

For connecting this autoencoder to the pipeline, I suggest looking at `encode_laser.py`. 