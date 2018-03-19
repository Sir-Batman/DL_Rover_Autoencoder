3 Layer Model (8 output)
===

Model
---
`laser_cae.py`contains the model

Parameters
---
`conv_autoencoder_3layer.pth` contains the learned parameters.

Logs
---
`cae_training_3layer.log` contains the loss as its being trained.


Comments
---

In this autoencoder, we encode 2 channels of 360 laser data to a 10 vector using 3 convolutional layers.

It was trained on 100,000 laser samples.

This is a second iteration of our model. It does not seem to work very well, as the decoded images don't seem to match the original.