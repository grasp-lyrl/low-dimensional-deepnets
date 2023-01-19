from networks.net_utils import * 

class random(nn.Module):
    def __init__(self, num_classes, class_probs=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_probs = class_probs or [1/num_classes]*num_classes
        self.class_probs = th.Tensor(self.class_probs)

    def forward(self, x):
        labels = th.multinomial(input=self.class_probs, num_samples=x.size(0), replacement=True)
        return nn.functional.one_hot(labels, num_classes=self.num_classes)
