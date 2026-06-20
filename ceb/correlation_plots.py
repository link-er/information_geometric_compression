import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
from pathlib import Path

betas = ['1000', '500', '100', '50', '10', '5', '2']

if __name__ == "__main__":
    EXP_PATH = Path('checkpoints/wrn28-4_cifar10/')

    cmap = matplotlib.colormaps['viridis']
    colors = cmap(np.linspace(0.1, 0.9, len(betas)))

    exp_data = []
    for exp in EXP_PATH.iterdir():
        if Path.exists(exp/"characteristics.pkl"):
            exp_data.append(pickle.load(open(exp/"characteristics.pkl", "rb")))
            cur_beta = str(exp).split("/")[-1].split('_')[0][4:]
            exp_data[-1]['color'] = colors[betas.index(cur_beta)]
            exp_data[-1]['beta'] = cur_beta

    gener_loss, gener_acc = [], []
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_mi_xz, test_mi_xz = [], []
    train_mi_yz, test_mi_yz = [], []
    train_g1, train_g2, train_g3 = [], [], []
    test_g1, test_g2, test_g3 = [], [], []
    backw_enc_NC = []
    beta_color, ebeta = [], []
    for e in exp_data:
        train_loss.append(e['train_class_loss'])
        test_loss.append(e['test_class_loss'])
        train_acc.append(e['train_avg_acc'])
        test_acc.append(e['test_avg_acc'])
        gener_loss.append(e['test_class_loss'] - e['train_class_loss'])
        gener_acc.append(e['train_avg_acc'] - e['test_avg_acc'])
        train_mi_xz.append(e['train_IXZ'])
        test_mi_xz.append(e['test_IXZ'])
        train_mi_yz.append(e['train_IYZ'])
        test_mi_yz.append(e['test_IYZ'])
        train_g1.append(e['train_NC_g1'])
        train_g2.append(e['train_H_bin_Z_g2'])
        train_g3.append(e['train_silh_sc'])
        test_g1.append(e['test_NC_g1'])
        test_g2.append(e['test_H_bin_Z_g2'])
        test_g3.append(e['test_silh_sc'])
        backw_enc_NC.append(e['backward_NC'])
        beta_color.append(e['color'])
        ebeta.append(float(e['beta']))

    # performance
    ax1 = plt.subplot(211)
    ax1.title.set_text('Train CE loss')
    for i in range(len(exp_data)):
        plt.scatter(ebeta[i], train_loss[i], color=beta_color[i])
    plt.tick_params('x', labelsize=6)
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.title.set_text('Test CE loss')
    for i in range(len(exp_data)):
        plt.scatter(ebeta[i], test_loss[i], color=beta_color[i])
    # make these tick labels invisible
    plt.tick_params('x', labelbottom=False)
    plt.xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "performance_loss.png", dpi=200)
    plt.close()

    ax1 = plt.subplot(211)
    ax1.title.set_text('Train acc')
    for i in range(len(exp_data)):
        plt.scatter(ebeta[i], train_acc[i], color=beta_color[i])
    plt.tick_params('x', labelsize=6)
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.title.set_text('Test acc')
    for i in range(len(exp_data)):
        plt.scatter(ebeta[i], test_acc[i], color=beta_color[i])
    # make these tick labels invisible
    plt.tick_params('x', labelbottom=False)
    plt.xscale('log')
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "performance_acc.png", dpi=200)
    plt.close()

    # generalization against MI(X,Z|Y)
    for i in range(len(exp_data)):
        plt.scatter(gener_loss[i], train_mi_xz[i], color = beta_color[i])
    plt.ylabel("training I(X;Z|Y)")
    plt.xlabel("generalization gap (loss)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "mi_gen_loss.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], train_mi_xz[i], color = beta_color[i])
    plt.ylabel("training I(X;Z|Y)")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "mi_gen_acc.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_loss[i], test_mi_xz[i], color = beta_color[i])
    plt.ylabel("test I(X;Z|Y)")
    plt.xlabel("generalization gap (loss)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_mi_gen_loss.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(test_acc[i], test_mi_xz[i], color = beta_color[i])
    plt.ylabel("test I(X;Z|Y)")
    plt.xlabel("test accuracy")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_mi_acc.png", dpi=200)
    plt.close()

    # generalization against geom compression
    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], train_g1[i], color = beta_color[i])
    plt.ylabel("train neural collapse measure")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_NC_gen.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], test_g1[i], color = beta_color[i])
    plt.ylabel("test neural collapse measure")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_NC_gen.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], backw_enc_NC[i], color = beta_color[i])
    plt.ylabel("backward encoder neural collapse")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "backw_NC_gen.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], train_g2[i], color = beta_color[i])
    plt.ylabel("train binned entropy")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_H_gen.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], test_g2[i], color = beta_color[i])
    plt.ylabel("test binned entropy")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_H_gen.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], train_g3[i], color = beta_color[i])
    plt.ylabel("train Silhouette score")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_clust_gen.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(gener_acc[i], test_g3[i], color = beta_color[i])
    plt.ylabel("test Silhouette score")
    plt.xlabel("generalization gap (accuracy)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_clust_gen.png", dpi=200)
    plt.close()

    # geom compression against MI(X,Z)
    for i in range(len(exp_data)):
        plt.scatter(train_mi_xz[i], train_g1[i], color = beta_color[i])
    plt.ylabel("train neural collapse measure")
    plt.xlabel("training I(X;Z|Y)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_NC_mi.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(test_mi_xz[i], test_g1[i], color = beta_color[i])
    plt.ylabel("test neural collapse measure")
    plt.xlabel("test I(X;Z|Y)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_NC_mi.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(train_mi_xz[i], train_g2[i], color = beta_color[i])
    plt.ylabel("train binned entropy")
    plt.xlabel("training I(X;Z|Y)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_H_mi.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(test_mi_xz[i], test_g2[i], color = beta_color[i])
    plt.ylabel("test binned entropy")
    plt.xlabel("test I(X;Z|Y)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_H_mi.png", dpi=200)
    plt.close()

    for i in range(len(exp_data)):
        plt.scatter(train_mi_xz[i], train_g3[i], color = beta_color[i])
    plt.ylabel("train Silhouette score")
    plt.xlabel("training I(X;Z|Y)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "train_clust_mi.png", dpi=200)
    plt.close()
    for i in range(len(exp_data)):
        plt.scatter(test_mi_xz[i], test_g3[i], color = beta_color[i])
    plt.ylabel("test Silhouette score")
    plt.xlabel("test I(X;Z|Y)")
    plt.tight_layout()
    # plt.show()
    plt.savefig(EXP_PATH / "test_clust_mi.png", dpi=200)
    plt.close()

