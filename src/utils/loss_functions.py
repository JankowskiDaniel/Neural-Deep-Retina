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


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input, target):
        input = input.view(-1)
        target = target.view(-1)

        tp = (input * target).sum()
        fp = ((1 - target) * input).sum()
        fn = (target * (1 - input)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)  # noqa: E501
        return 1 - tversky
