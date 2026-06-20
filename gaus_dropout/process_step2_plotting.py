import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from pathlib import Path

lmbds = ['0.001', '0.005', '0.01', '0.05', '0.1', '0.5', '1.0', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5', '10']

if __name__ == "__main__":
    EXP_PATH = Path('checkpoints/densenet_cifar100_v1/')

    cmap = matplotlib.colormaps['viridis']
    colors = cmap(np.linspace(0.1, 0.9, len(lmbds)))

    exp_data = []
    for exp in EXP_PATH.iterdir():
        if Path.exists(exp/"characteristics.pkl"):
            exp_data.append(pickle.load(open(exp/"characteristics.pkl", "rb")))
            cur_lmbd = str(exp).split("\\")[-1].split('_')[0][4:]
            exp_data[-1]['color'] = colors[lmbds.index(cur_lmbd)]
            exp_data[-1]['lmbd'] = cur_lmbd

    gener_loss, gener_acc = [], []
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_mi_xz, test_mi_xz = [], []
    #train_g1, train_g2, train_g3 = [], [], []
    #test_g1, test_g2, test_g3 = [], [], []
    train_g1 = []
    test_g1 = []
    lmbd_color, elmbd = [], []
    for e in exp_data:
        train_loss.append(e['train_loss'])
        test_loss.append(e['test_loss'])
        train_acc.append(e['train_acc'])
        test_acc.append(e['test_acc'])
        gener_loss.append(e['test_loss'] - e['train_loss'])
        gener_acc.append(e['train_acc'] - e['test_acc'])
        train_mi_xz.append(e['train_IXZ'])
        test_mi_xz.append(e['test_IXZ'])
        train_g1.append(e['train_NC_g1'])
        #train_g2.append(e['train_H_bin_Z_g2'])
        #train_g3.append(e['train_silh_sc'])
        test_g1.append(e['test_NC_g1'])
        #test_g2.append(e['test_H_bin_Z_g2'])
        #test_g3.append(e['test_silh_sc'])
        lmbd_color.append(e['color'])
        elmbd.append(float(e['lmbd']))

    # performance
    ax1 = plt.subplot(211)
    ax1.title.set_text('Train loss')
    for i in range(len(exp_data)):
        plt.scatter(elmbd[i], train_loss[i], color=lmbd_color[i])
    plt.tick_params('x', labelsize=6)
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.title.set_text('Test loss')
    for i in range(len(exp_data)):
        plt.scatter(elmbd[i], test_loss[i], color=lmbd_color[i])
    # make these tick labels invisible
    plt.tick_params('x', labelbottom=False)
    #plt.xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "performance_loss.png", dpi=200)
    plt.close()

    ax1 = plt.subplot(211)
    ax1.title.set_text('Train acc')
    for i in range(len(exp_data)):
        plt.scatter(elmbd[i], train_acc[i], color=lmbd_color[i])
    plt.tick_params('x', labelsize=6)
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.title.set_text('Test acc')
    for i in range(len(exp_data)):
        plt.scatter(elmbd[i], test_acc[i], color=lmbd_color[i])
    # make these tick labels invisible
    plt.tick_params('x', labelbottom=False)
    #plt.xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "performance_acc.png", dpi=200)
    plt.close()

    # generalization against MI(X,Z|Y)
    for i in range(len(exp_data)):
        plt.scatter(gener_loss[i], train_mi_xz[i], color = lmbd_color[i])
    plt.ylabel("training I(X;Z)")
    plt.xlabel("generalization gap (loss)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "mi_gen_loss.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], train_mi_xz[i], color = lmbd_color[i])
    plt.ylabel("training I(X;Z)")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "mi_gen_acc.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_loss[i], test_mi_xz[i], color = lmbd_color[i])
    plt.ylabel("test I(X;Z)")
    plt.xlabel("generalization gap (loss)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_mi_gen_loss.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(test_acc[i], test_mi_xz[i], color = lmbd_color[i])
    plt.ylabel("test I(X;Z)")
    plt.xlabel("test accuracy")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_mi_acc.png", dpi=200)
    plt.close()

    # generalization against geom compression
    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], train_g1[i], color = lmbd_color[i])
    plt.ylabel("train neural collapse measure")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_NC_gen.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], test_g1[i], color = lmbd_color[i])
    plt.ylabel("test neural collapse measure")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_NC_gen.png", dpi=200)
    plt.close()

    # geom compression against MI(X,Z)
    for i in range(len(exp_data)):
        plt.scatter(train_mi_xz[i], train_g1[i], color = lmbd_color[i])
    plt.ylabel("train neural collapse measure")
    plt.xlabel("training I(X;Z)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_NC_mi.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(test_mi_xz[i], test_g1[i], color = lmbd_color[i])
    plt.ylabel("test neural collapse measure")
    plt.xlabel("test I(X;Z)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_NC_mi.png", dpi=200)
    plt.close()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(exp_data)):
        ax.scatter(train_mi_xz[i], train_g1[i], gener_acc[i], color = lmbd_color[i])
    ax.set_xlabel("train I(X;Z)")
    ax.set_ylabel("training NC")
    ax.set_zlabel("generalization gap (acc)")
    plt.show()

