import numpy as np
from scipy.stats import rankdata, pearsonr
import pickle
from pathlib import Path

exps = ['densenet_cifar100_v1'] #'bert_news', 'vgg11_svhn_overfitting_pattern', , 'resnet18_cifar10', 'vgg11_svhn']

def modify_into_ranks(values):
    ranks = rankdata(values)
    # normalize by min-max scaling to [0,1]
    return (ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))

def print_pretty(correlation, signif, name1, name2):
    print("Correlation between", name1, "and", name2, "is", str(round(correlation, 4)), " with significance ",
          str(round(signif, 6)))

if __name__ == "__main__":
    # collect every experiment data and modify into ranks
    # concatenate all of them together and compute correlation coefficient
    exp_lmbds = []
    train_accs, test_accs = [], []
    train_mi, test_mi = [], []
    cond_train_mi, cond_test_mi = [], []
    train_NC, test_NC = [], []
    train_sihl, test_sihl = [], []
    train_entr, test_entr = [], []
    gener_accs = []
    for setup in exps:
        EXP_PATH = Path('checkpoints/' + setup) # + '/results_summary/')

        exp_data = []
        for exp in EXP_PATH.iterdir():
            if Path.exists(exp/"characteristics.pkl"):
                exp_data.append(pickle.load(open(exp/"characteristics.pkl", "rb")))
                cur_lmbd = str(exp).split("\\")[-1].split('_')[0][4:]
                exp_data[-1]['lmbd'] = cur_lmbd

        gener_loss, gener_acc = [], []
        train_loss, test_loss = [], []
        train_acc, test_acc = [], []
        train_mi_xz, test_mi_xz = [], []
        cond_train_mi_xz, cond_test_mi_xz = [], []
        train_g1, train_g2, train_g3 = [], [], []
        test_g1, test_g2, test_g3 = [], [], []
        #backw_enc_NC = []
        elmbd = []
        for e in exp_data:
            train_loss.append(e['train_loss'])
            test_loss.append(e['test_loss'])
            train_acc.append(e['train_acc'])
            test_acc.append(e['test_acc'])
            gener_loss.append(e['test_loss'] - e['train_loss'])
            gener_acc.append(e['train_acc'] - e['test_acc'])
            train_mi_xz.append(e['train_IXZ'])
            test_mi_xz.append(e['test_IXZ'])
            cond_train_mi_xz.append(e['train_IXZ_givenY'])
            cond_test_mi_xz.append(e['test_IXZ_givenY'])
            train_g1.append(e['train_NC_g1'])
            #train_g2.append(e['train_H_bin_Z_g2'])
            #train_g3.append(e['train_silh_sc'])
            test_g1.append(e['test_NC_g1'])
            #test_g2.append(e['test_H_bin_Z_g2'])
            #test_g3.append(e['test_silh_sc'])
            #backw_enc_NC.append(e['backward_NC'])
            elmbd.append(float(e['lmbd']))

        exp_lmbds += modify_into_ranks(elmbd).tolist()
        train_accs += modify_into_ranks(train_acc).tolist()
        train_mi += modify_into_ranks(train_mi_xz).tolist()
        cond_train_mi += modify_into_ranks(cond_train_mi_xz).tolist()
        train_NC += modify_into_ranks(train_g1).tolist()
        #train_sihl += modify_into_ranks(train_g3).tolist()
        #train_entr += modify_into_ranks(train_g2).tolist()

        gener_accs += modify_into_ranks(gener_acc).tolist()

        test_accs += modify_into_ranks(test_acc).tolist()
        test_mi += modify_into_ranks(test_mi_xz).tolist()
        cond_test_mi += modify_into_ranks(cond_test_mi_xz).tolist()
        test_NC += modify_into_ranks(test_g1).tolist()
        #test_sihl += modify_into_ranks(test_g3).tolist()
        #test_entr += modify_into_ranks(test_g2).tolist()

    print("Train data")
    print_pretty(*pearsonr(train_accs, exp_lmbds), "accuracy", "lambda")
    print_pretty(*pearsonr(cond_train_mi, exp_lmbds), "I(X,Z|Y)", "lambda")
    print_pretty(*pearsonr(train_mi, exp_lmbds), "I(X,Z)", "lambda")
    print_pretty(*pearsonr(train_NC, exp_lmbds), "NC", "lambda")
    #r, p_value = pearsonr(train_sihl, exp_lmbds)
    #r, p_value = pearsonr(train_entr, exp_lmbds)

    print_pretty(*pearsonr(cond_train_mi, train_NC), "I(X,Z|Y)", "NC")
    print_pretty(*pearsonr(train_mi, train_NC), "I(X,Z)", "NC")
    #r, p_value = pearsonr(cond_train_mi, train_sihl)
    #r, p_value = pearsonr(cond_train_mi, train_entr)

    print_pretty(*pearsonr(train_accs, cond_train_mi), "accuracy", "I(X,Z|Y)")
    print_pretty(*pearsonr(train_accs, train_mi), "accuracy", "I(X,Z)")
    print_pretty(*pearsonr(train_accs, train_NC), "accuracy", "NC")

    print("Generalization")
    print_pretty(*pearsonr(gener_accs, exp_lmbds), "generalization gap (acc)", "lambda")
    print_pretty(*pearsonr(gener_accs, cond_train_mi), "generalization gap (acc)", "train I(X,Z|Y)")
    print_pretty(*pearsonr(gener_accs, train_mi), "generalization gap (acc)", "train I(X,Z)")
    print_pretty(*pearsonr(gener_accs, train_NC), "generalization gap (acc)", "train NC")
    print_pretty(*pearsonr(gener_accs, cond_test_mi), "generalization gap (acc)", "test I(X,Z|Y)")
    print_pretty(*pearsonr(gener_accs, test_mi), "generalization gap (acc)", "test I(X,Z)")
    print_pretty(*pearsonr(gener_accs, test_NC), "generalization gap (acc)", "test NC")

    print("Test data")
    print_pretty(*pearsonr(test_accs, exp_lmbds), "accuracy", "lambda")
    print_pretty(*pearsonr(cond_test_mi, exp_lmbds), "I(X,Z|Y)", "lambda")
    print_pretty(*pearsonr(test_mi, exp_lmbds), "I(X,Z)", "lambda")
    print_pretty(*pearsonr(test_NC, exp_lmbds), "NC", "lambda")
    #r, p_value = pearsonr(test_sihl, exp_lmbds)
    #r, p_value = pearsonr(test_entr, exp_lmbds)

    print_pretty(*pearsonr(cond_test_mi, test_NC), "I(X,Z|Y)", "NC")
    print_pretty(*pearsonr(test_mi, test_NC), "I(X,Z)", "NC")
    #r, p_value = pearsonr(cond_test_mi, test_sihl)
    #r, p_value = pearsonr(cond_test_mi, test_entr)

    print_pretty(*pearsonr(test_accs, cond_test_mi), "accuracy", "I(X,Z|Y)")
    print_pretty(*pearsonr(test_accs, test_mi), "accuracy", "I(X,Z)")
    print_pretty(*pearsonr(test_accs, test_NC), "accuracy", "NC")

