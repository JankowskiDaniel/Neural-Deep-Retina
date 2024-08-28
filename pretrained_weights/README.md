# Pre-trained weights

This folder is intended to store pre-trained weights of PyTorch models used in the pipeline.

By default, PyTorch saves downloaded models in `$TORCH_HOME/models`.
In order to have more control over the models used and disk space, we download the weights manually and save in this location.

The weight download urls can be found at https://pytorch.org/vision/stable/models.html (you need to go to source).

Once found, the weights can be downloaded for example by wget:
```sh
wget -c https://download.pytorch.org/models/vgg16-397923af.pth
```

Please put the downloaded weights in this folder, so that they can be loaded smoothly.

### Model Zoo

Here, we list the available models.
Please mind that to use the concrete model, you need to provide the correct values in the `config` file.
The class name is provided in parentheses.

For example:
```yaml
TRAINING:
  ENCODER:
    name: CustomEncoder # class name
    weights: custom_encoder_best.pth # weights file name
```

#### VGG 16 (VGG16Encoder)

```sh
wget -c https://download.pytorch.org/models/vgg16-397923af.pth
```

#### Video classification MC3 18 ResNet (MC3VideoEncoder)

Important: for now, the weights for MC3 are downloaded automatically (no need for manual downloading).

```sh
wget -c https://download.pytorch.org/models/mc3_18-a90a0ba3.pth
```

More info about the network here: https://pytorch.org/vision/master/models/video_resnet.html.

#### Custom encoder trained from scratch (CustomEncoder)

https://drive.google.com/file/d/14Ws9qtM0lRmCKpHAm82MSubH6sm5cvI4/view?usp=drive_link

