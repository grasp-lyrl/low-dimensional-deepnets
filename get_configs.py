import yaml
for bs in [200, 500, 1000]:
    for lr in [0.001, 5e-4]:
        lr = lr/200*bs
        for wd in [0.0, 0.001, 1e-5]:
            config = {'autocast': True, 'batch_seed': -1, 'bs': bs, 'epochs': 200, 'opt': 'Adam',
                      'opt_args': {'lr': lr, 'weight_decay': wd}, 'sched_args': None, 'scheduler': 'cosine'}
            if bs > 200 and wd == 0.001:
                model = '-vit'
            elif bs > 200:
                model = '-all'
            else:
                model = ''
            fname = f'configs/optim/adam-{bs}-{lr}-{wd}{model}.yaml'
            with open(fname, 'w') as f:
                yaml.safe_dump(config, f)
