3 Layer Model (8 output)

Model
---
`laser_cae.py`contains the model

Parameters
---
`conv_autoencoder.pth` contains the learned parameters.

Logs
---
`train_std.log` contains the stdout while training, just for logging purposes.
`train.csv` contains the losses while being trained (epoch, train_loss, test_loss)
`encode_sample.log` contains a sample encoding and decoding using this autoencoder


Comments
---

In this autoencoder, we encode 2 channels of 360 laser data to a 10 vector using 3 convolutional layers.
Removed Batch Norm Layers from model 1, 
Removed Max pool layer and do purely convolutions.

It was trained on 100,000 laser samples.



