# Low-Dimensional-Deepnets
Code to reproduce results of the paper [The training process of many deep networks explores the same low-dimensional manifold](https://arxiv.org/abs/2210.17011) published in Proceedings of Natural Academy of Science (PNAS), vol 121. No. 12, 2024.

## Usage
`notebooks/all_plots.ipynb` contains the code to reproduce most of the figures in the main paper and the appendix. Tangent vector embeddings shown in Fig. 10, 11 can be reproduced using code in `notebooks/tangents.ipynb`. Joint embedding of train and test data as in Fig. 6 can be reproduced by `notebook/subset_embedding.ipynb`. Corner experiments in Appendix E.1 can be reproduced by `notebooks/corner.ipynb`. Analyses of different averaging methods in Appendix E.4 can be reproduced by `notebooks/mean_traj_plots.ipynb`. We also provide interactive versions of Fig. 2 and Fig. 5 in the paper as html files in `notebooks/CIFAR_train.html` and `notebooks/CIFAR_test.html`.

## Data
Data necessary for reproducing plots in the main paper (including training configuration data, projected coordinates and pairwise Bhattacharya distances) are available on [aws s3](https://jmao-penn.s3.amazonaws.com/inpca_results_all/). Notice the pairwise distances matrix has lower precision than the one used for the analyses in the paper. Full precision distance matrices, raw outputs of all the trained models, and data used for plots in the appendix are available upon request.
