# Baseline Data Handlers #

This module contains data handlers for the baseline model. A subsequence of images is created already in the convolutional stage. Three consecutive images are concatenated into a single image (each image is parsed into a separate color channel). Both non-sequential and sequential versions were implemented. In the case of sequential version, the data parsed to the model is redundant, but this is not a problem.