from pathlib import Path
import pickle
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from datasets.datasets import *
from models import *
from collections import Counter

def compute_decision_boundaries(x_min, x_max, y_min, y_max, predictor):
    # Create a grid of points with a specified resolution
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max-x_min)/200.0), np.arange(y_min, y_max, (y_max-y_min)/200.0))
    # Stack the grid to create a list of (x, y) points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    # Predict using the neural network
    preds = predictor(torch.tensor(grid_points, dtype=torch.float).cuda())
    Z = preds.max(1)[1].cpu().data.numpy()
    # Reshape the predictions to match the grid shape
    Z = Z.reshape(xx.shape)
    return xx, yy, Z

if __name__=="__main__":
    EXP_PATH_parent = Path('checkpoints/vis_fc_FashionMNIST')

    exp_data = []
    for exp in EXP_PATH_parent.iterdir():
        EXP_PATH = exp
        print(exp)
        if not '10_seed7' in str(exp):
            continue
        args = {}
        args['beta'] = int(str(exp).split('\\')[-1].split('_')[0][4:])

        # not very clean way for sharing and saving setup of the experiment
        exec(open(EXP_PATH / "params.py").read())

        #cmap = matplotlib.colormaps['tab20']
        #colors = cmap(np.linspace(0.05, 0.95, num_classes))
        colors = [
            'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple',
            'magenta', 'grey', 'brown'
        ]

        for ep in [10, 150]:
            model.load_state_dict(torch.load(EXP_PATH / ("chkp_" + str(ep))))
            model.eval()
            # collect indices of different class samples for geometric compression measures
            train_class_inds, test_class_inds = {}, {}
            for c in range(num_classes):
                train_class_inds[c], test_class_inds[c] = [], []
            train_reprs = []
            for ind, d in enumerate(data_loaders['train']):
                cur_reprs = model.representation(d[0].cuda())
                train_reprs += cur_reprs.data.cpu().numpy().tolist()
                for c in range(num_classes):
                    train_class_inds[c] += ((d[1] == c).nonzero().flatten() + ind * batch_size).tolist()
            train_reprs = np.array(train_reprs)
            test_reprs = []
            for ind, d in enumerate(data_loaders['test']):
                test_reprs += model.representation(d[0].cuda()).data.cpu().numpy().tolist()
                for c in range(num_classes):
                    test_class_inds[c] += ((d[1] == c).nonzero().flatten() + ind * batch_size).tolist()
            test_reprs = np.array(test_reprs)

            x_min, x_max = train_reprs[:, 0].min() - 1, train_reprs[:, 0].max() + 1
            y_min, y_max = train_reprs[:, 1].min() - 1, train_reprs[:, 1].max() + 1
            xx, yy, Z = compute_decision_boundaries(x_min, x_max, y_min, y_max, model.decoder)

            plt.rcParams.update({'figure.figsize': (10, 10)})
            plt.rcParams.update({'font.size': 15})
            #plt.contourf(xx, yy, Z, alpha=0.5, cmap=matplotlib.colors.ListedColormap(colors))
            plt.scatter(xx, yy, c=Z, cmap=matplotlib.colors.ListedColormap(colors), marker='s', edgecolors=None, alpha=0.1)
            for i in range(num_classes):
                #plt.contourf(xx, yy, (Z == i), alpha=0.5, colors=[colors[i]])
                plt.scatter(train_reprs[train_class_inds[i]][:, 0], train_reprs[train_class_inds[i]][:, 1],
                            color=colors[i], label='Class '+str(i), s=10)
            plt.legend()
            plt.show()

            plt.scatter(xx, yy, c=Z, cmap=matplotlib.colors.ListedColormap(colors), marker='s', edgecolors=None, alpha=0.1)
            for i in range(num_classes):
                plt.scatter(test_reprs[test_class_inds[i]][:, 0], test_reprs[test_class_inds[i]][:, 1],
                            color=colors[i], label='Class '+str(i), s=10)
            plt.legend()
            plt.show()
