from torch import nn, mean


class SingnalWeightedMSELoss(nn.Module):
    def __init__(self, epsilon: float = 0.001):
        super(SingnalWeightedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return mean((input - target) ** 2 * (target + self.epsilon))
