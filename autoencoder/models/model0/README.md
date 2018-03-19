2 Layer Model (10 output)
===

Model
---
`laser_cae.py`contains the model

Parameters
---
`conv_autoencoder.pth` contains the learned parameters.

Logs
---
`cae_training.log` contains the loss as its being trained.
`train.log` shows initial/encode/decode on a laser scan from training set
`test.log` shows initial/encode/decode on a sample laser scan not from the training set


Comments
---

In this autoencoder, we encode 2 channels of 360 laser data to a 10 vector using 2 convolutional layers.
This is our initial model. It does not seem to work very well, as the decoded images don't seem to match the original.
