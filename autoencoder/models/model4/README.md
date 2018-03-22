Sparse Linear Autoencoder: 3 Layer Model (8 output)
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

In this autoencoder, we try using a straight up linear autoencoder without batch norm. We also add an L1 penalty to get a sparse encoding