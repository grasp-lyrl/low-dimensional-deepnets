import torch as th

def init_weights(m,  **kwargs):
    if isinstance(m, th.nn.Linear):
        th.nn.init.kaiming_normal_(
            m.weight, **kwargs)
