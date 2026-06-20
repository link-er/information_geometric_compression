import numpy as np
from scipy.stats import rankdata, pearsonr
import pickle
from pathlib import Path

exps = ['wrn28-4_cifar10', 'densenet121_cifar100', 'lenet_FashionMNIST', 'largeFC_FMNIST']

def modify_into_ranks(values):
    ranks = rankdata(values)
    # normalize by min-max scaling to [0,1]
    return (ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))

if __name__ == "__main__":
    # collect every experiment data and modify into ranks
    # concatenate all of them together and compute correlation coefficient
    exp_betas = []
    train_accs, test_accs = [], []
    train_mi, test_mi = [], []
    train_NC, test_NC = [], []
    train_sihl, test_sihl = [], []
    train_entr, test_entr = [], []
    gener_accs = []
    for setup in exps:
        EXP_PATH = Path('checkpoints/' + setup + '/results_summary/')

        exp_data = []
        for exp in EXP_PATH.iterdir():
            if Path.exists(exp/"characteristics.pkl"):
                exp_data.append(pickle.load(open(exp/"characteristics.pkl", "rb")))
                cur_beta = str(exp).split("\\")[-1].split('_')[0][4:]
                exp_data[-1]['beta'] = cur_beta

        gener_loss, gener_acc = [], []
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        train_mi_xz, test_mi_xz = [], []
        train_mi_yz, test_mi_yz = [], []
        train_g1, train_g2, train_g3 = [], [], []
        test_g1, test_g2, test_g3 = [], [], []
        #backw_enc_NC = []
        ebeta = []
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
            #backw_enc_NC.append(e['backward_NC'])
            ebeta.append(float(e['beta']))

        exp_betas += modify_into_ranks(ebeta).tolist()
        train_accs += modify_into_ranks(train_acc).tolist()
        train_mi += modify_into_ranks(train_mi_xz).tolist()
        train_NC += modify_into_ranks(train_g1).tolist()
        train_sihl += modify_into_ranks(train_g3).tolist()
        train_entr += modify_into_ranks(train_g2).tolist()

        gener_accs += modify_into_ranks(gener_acc).tolist()

        test_accs += modify_into_ranks(test_acc).tolist()
        test_mi += modify_into_ranks(test_mi_xz).tolist()
        test_NC += modify_into_ranks(test_g1).tolist()
        test_sihl += modify_into_ranks(test_g3).tolist()
        test_entr += modify_into_ranks(test_g2).tolist()

    print("Train data")
    r, p_value = pearsonr(train_accs, exp_betas)
    print("Correlation between accuracy and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_mi, exp_betas)
    print("Correlation between MI and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_NC, exp_betas)
    print("Correlation between NC and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_sihl, exp_betas)
    print("Correlation between Sihloette coefficient and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_entr, exp_betas)
    print("Correlation between binned entropy and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))

    r, p_value = pearsonr(train_mi, train_NC)
    print("Correlation between MI and NC is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_mi, train_sihl)
    print("Correlation between MI and Sihloette coefficient is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_mi, train_entr)
    print("Correlation between MI and binned entropy is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))

    r, p_value = pearsonr(train_accs, train_mi)
    print("Correlation between accuracy and MI is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(train_accs, train_NC)
    print("Correlation between accuracy and NC is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))

    print("Generalization")
    r, p_value = pearsonr(gener_accs, exp_betas)
    print("Correlation between generalization gap (acc) and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(gener_accs, train_mi)
    print("Correlation between generalization gap (acc) and MI is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(gener_accs, train_NC)
    print("Correlation between generalization gap (acc) and NC is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))

    print("Test data")
    r, p_value = pearsonr(test_accs, exp_betas)
    print("Correlation between accuracy and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_mi, exp_betas)
    print("Correlation between MI and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_NC, exp_betas)
    print("Correlation between NC and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_sihl, exp_betas)
    print("Correlation between Sihloette coefficient and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_entr, exp_betas)
    print("Correlation between binned entropy and beta is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))

    r, p_value = pearsonr(test_mi, test_NC)
    print("Correlation between MI and NC is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_mi, test_sihl)
    print("Correlation between MI and Sihloette coefficient is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_mi, test_entr)
    print("Correlation between MI and binned entropy is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))

    r, p_value = pearsonr(test_accs, test_mi)
    print("Correlation between accuracy and MI is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
    r, p_value = pearsonr(test_accs, test_NC)
    print("Correlation between accuracy and NC is ", str(round(r, 4)), " with significance ", str(round(p_value, 6)))
