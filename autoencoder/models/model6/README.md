3 Layer Model (8 output) - Clone of Model 1, used to generate data for presentation
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


Comments
---

In this autoencoder, we encode 2 channels of 360 laser data to a 10 vector using 3 convolutional layers.


It was trained on 100,000 laser samples.



