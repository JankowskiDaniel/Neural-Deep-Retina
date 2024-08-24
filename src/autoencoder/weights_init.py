from torch.nn import Conv2d
from torch.nn.init import xavier_uniform_, uniform_


def conv_weights_init(m):
    if isinstance(m, Conv2d):
        xavier_uniform_(m.weight.data)
        if m.bias is not None:
            uniform_(m.bias)
