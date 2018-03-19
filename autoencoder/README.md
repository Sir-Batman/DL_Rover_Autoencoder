AutoEncoder
===========

Scripts
----
`mnist_cae.py` - Initial sample autoencoder for MNIST that we adapted to work with laser data.

`laser_cae.py` - Convolutional Autoencoder used to encode laser data

`laser_ae.py` - Linear Autoencoder for laser data. Does not seem (?) work but retained for posterity.

`encode_laser.py` - A sample file showing how to load the autoencoder and call it on different laser scan data.

Data Files
----
`conv_autoencoder.pth` - Contains the trained model on 100,000 laser scans.

`cae_training.log` - The log of the loss as the model was trained.

`sample_poi_laser.csv` - Contains 10 POI laser scans used to sanity check the model (samples not used to train model)

`sample_rov_laser.csv` - Contains 10 ROV laser scans used to sanity check the model (samples not used to train model)

`train_poi_laser.csv` - Contains 10 POI laser scans used to sanity check the model (samples from 100,000 used to train model)

`train_rov_laser.csv` - Contains 10 ROV laser scans used to sanity check the model (samples from 100,000 used to train model)

`test.log` - example of input, encode, decode of first laser scan in the sample csv files.

`train.log` - example of input, encode, decode of first laser scan in the train csv files.

Spreadsheet
----
`layer_size_calc.ods` - spreadsheet that computes the output size of each layer given parameters.

Current Status
----
It seems that the autoencoder's performance does not work very well when given 100,000 damples to train on. Looking at `test.log` and `train.log`, the decoded laser scan does not seem very accurate. More development of the network is necessary, but this may be enough for the presentation.

For connecting this autoencoder to the pipeline, I suggest looking at encode_laser.py. 