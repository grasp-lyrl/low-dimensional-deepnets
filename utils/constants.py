CHOICES = {
    'seed': range(42, 52),
    'm': ["allcnn", "convmixer", "fc",
          "vit", "wr-10-4-8", "wr-16-4-64"],
    'opt': ["adam", "sgd", "sgdn"],
    'corner': ["uniform", "normal", "subsample-2000", "subsample-200"],
    'lr': [0.001, 0.1, 0.25, 0.5, 0.0005, 0.0025, 0.00125,
           0.005],
    'bs': [200, 500],
    'aug': ['simple', 'none'],
    'wd': [0., 1.e-03, 1.e-05]
}

CDICT_M = {'allcnn': '#e41a1c',
            'convmixer': '#377eb8',
            'fc': '#4daf4a',
            'vit': '#984ea3',
            'wr-10-4-8': '#ff7f00',
            'wr-16-4-64': '#ffff33',
            'geodesic': '#a65628'}