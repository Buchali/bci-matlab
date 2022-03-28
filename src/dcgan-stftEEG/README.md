# dcgan-stftEEG

This project aims to classify MI tasks using Deep Convolutional Generative Adversial Network based on stft images of EEG signals.

First, run `preprocess.m` to create image data. Only run this the first time to convert .mat data to imageDatastore.

Then, run `model.m`.

*This is a rough implementation of the following article:*

===========================================================================

Zhang, K., Xu, G., Han, Z., Ma, K., Zheng, X., Chen, L., ... & Zhang, S. (2020). Data augmentation for motor imagery signal classification based on a hybrid neural network. Sensors, 20(16), 4485.

===========================================================================
