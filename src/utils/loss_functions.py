from torch import nn, mean, ones_like


class SingnalWeightedMSELoss(nn.Module):
    def __init__(self, epsilon: float = 0.001):
        super(SingnalWeightedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return mean((input - target) ** 2 * (target + self.epsilon))


class FrequencyWeightedMSELoss(nn.Module):
    def __init__(self, threshold: float = 1.3):
        super(FrequencyWeightedMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):
        weights = ones_like(target)
        weights[target >= self.threshold] = 5
        return (weights * (input - target) ** 2).sum() / weights.sum()
