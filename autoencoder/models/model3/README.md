Linear Autoencoder: 3 Layer Model (8 output)
===

Model
---
`laser_ae.py`contains the model

Parameters
---
`conv_autoencoder.pth` contains the learned parameters.

Logs
---


Comments
---

In this autoencoder, we try using a straight up linear autoencoder. Unfortunately batch norm is causing issues with the encode/decode sampler. In model 4, we do the same model but without the batch norm layer.