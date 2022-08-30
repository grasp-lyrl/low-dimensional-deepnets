CHOICES = {
    'm': ["allcnn", "convmixer", "fc",
          "vit", "wr-10-4-8", "wr-16-4-64"],
    'opt': ["adam", "sgd", "sgdn"],
    'lr': [0.001, 0.1, 0.25, 0.5, 0.0005, 0.0025, 0.00125,
           0.005],
    'bs': [200, 500],
    'aug': ['simple', 'none'],
    'wd': [0., 1.e-03, 1.e-05]
}
