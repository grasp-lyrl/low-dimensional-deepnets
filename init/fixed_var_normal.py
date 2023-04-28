import torch as th

def init_weights(m, mean=0.0, std=1.0):
    if isinstance(m, th.nn.Linear):
        th.nn.init.normal_(m.weight, mean=mean, std=std)
