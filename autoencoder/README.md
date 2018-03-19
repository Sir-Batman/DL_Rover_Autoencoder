AutoEncoder
===========

Scripts
----
`train_cae.py` - Trains the Convolutional Autoencoder. Select which encoder to use with the `model_file` parameter

`encode_laser.py` - A sample file showing how to load the autoencoder and call it on different laser scan data.

`LaserDataset.py` - LaserDataset class used to load laser data


Models
----
The `models` folder contains various iterations of our autoencoder. Read the README for each one to get a sense of what they are.

Each model in the model folder contains the Autoencoder Class in `laser_cae.py` and the model parameters in `conv_autoencoder.pth`.

It also contains some log files for how the encoder was trained.

`model0` - Our initial autoencoder that encodes to 10 vector. Still poor decode performance.

`model1` - First iteration. Now encodes to 8 vector. Still poor decode performance.

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