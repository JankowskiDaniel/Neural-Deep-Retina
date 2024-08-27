# Pre-trained weights

This folder is intended to store pre-trained weights of PyTorch models used in the pipeline.

By default, PyTorch saves downloaded models in `$TORCH_HOME/models`.
In order to have more control over the models used and disk space, we download the weights manually and save in this location.

The weight download urls can be found at https://pytorch.org/vision/stable/models.html (you need to go to source).

Once found, the weights can be downloaded for example by wget:
```sh
wget -c https://download.pytorch.org/models/vgg16-397923af.pth
```

For the video classification pre-trained MC3 18 ResNet use:

```sh
wget -c https://download.pytorch.org/models/mc3_18-a90a0ba3.pth
```

More info about the network here: https://pytorch.org/vision/master/models/video_resnet.html.

Please put the downloaded weights in this folder, so that they can be loaded smoothly.