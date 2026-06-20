import numpy as np
import torch
import pickle
import argparse
import importlib.util
from pathlib import Path

from NC_regularizer import compute_cluster

def validation(model, criterion, test_loader, llm_mode=False, device='cuda'):
    model.eval()
    loss_record = 0.0
    acc_record = 0
    with torch.no_grad():
        for B in test_loader:
            if llm_mode:
                input_ids = B["input_ids"].to(device)
                attention_mask = B["attention_mask"].to(device)
                y_batch = B["label"].to(device)
                y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                (X_batch, y_batch) = B
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch).item()
            loss_record += loss
            preds = y_pred.argmax(dim=1)
            acc = (preds == y_batch).float().mean().item()
            acc_record += acc
        num_batches = len(test_loader)
        loss_record /= num_batches
        acc_record /= num_batches
    return loss_record, acc_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', dest='directory', type=str, help='directory')
    parser.add_argument('--lmbd', dest='lmbd', type=float, help='orthogonalization strength')
    parser.add_argument('--seed', dest='seed', type=int, help='random seed')
    parser.add_argument('--val', dest='val', type=float, help='validation split, for tuning', default=0)
    parser.add_argument('--llm_mode', dest='llm_mode', type=str, help='is it a text transformer being trained?', default='False')
    args = parser.parse_args()
    args.llm_mode = args.llm_mode.lower() == "true"
    EXP_PATH = Path(args.directory)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    spec = importlib.util.spec_from_file_location("param_setup", (EXP_PATH / "params.py"))
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)

    criterion = torch.nn.CrossEntropyLoss()

    if args.val > 0:
        check_dataloader = params.data_loaders['val']
    else:
        check_dataloader = params.data_loaders['test']

    # Get original loss:
    print("-------Initialization----------")
    loss, acc = validation(params.model, criterion, check_dataloader, llm_mode=args.llm_mode)
    print("Loss", round(loss, 4))

    train_history = {}
    test_history = {}
    cdnv_history = {}

    # Training:
    for i in range(params.epochs):
        cdnv_vals = []
        params.model.train()
        loss_record = 0.0
        acc_record = 0
        for B in params.data_loaders['train']:
            params.optimizer.zero_grad()
            if args.llm_mode:
                input_ids = B["input_ids"].cuda()
                attention_mask = B["attention_mask"].cuda()
                y_batch = B["label"].cuda()
                outputs, second_last_outputs = params.model(input_ids=input_ids, attention_mask=attention_mask, return_repr=True)
            else:
                X_batch, y_batch = B
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
                outputs, second_last_outputs = params.model(X_batch, return_repr=True)
            preds = outputs.argmax(dim=1)
            acc = (preds == y_batch).float().mean().item()
            acc_record += acc
            loss = criterion(outputs, y_batch)
            if args.lmbd > 0 or args.lmbd < 0:
                cluster_metric, _, _ = compute_cluster(second_last_outputs, y_batch)
            else:
                cluster_metric = torch.tensor(0.0).cuda()
            cdnv_vals.append(float(cluster_metric))
            # ensure finite
            cluster_metric = torch.nan_to_num(cluster_metric, nan=0.0, posinf=1e6, neginf=-1e6)
            # we are forcing clusters to be less compact, i.e., preventing NC
            #print(loss.item(), args.lmbd, torch.tanh(beta*cluster_metric).item())
            loss = loss - args.lmbd * torch.tanh(params.beta*cluster_metric)
            loss_record += loss.item()
            loss.backward()
            params.optimizer.step()
        params.scheduler.step()
        cdnv_vals = np.array(cdnv_vals)
        print("CDNV stats: mean {:.3e}, median {:.3e}, min {:.3e}, max {:.3e}".format(cdnv_vals.mean(),
                                                                                      np.median(cdnv_vals),
                                                                                      cdnv_vals.min(),
                                                                                      cdnv_vals.max()))

        num_batches = len(params.data_loaders['train'])
        loss_record /= num_batches
        acc_record /= num_batches
        print("Epoch", i+1)
        print("Training loss", round(loss_record, 4), "Training accuracy", round(acc_record, 4))

        if i == 0:
            train_history = {'loss': [loss_record], 'acc': [acc_record]}
            train_history["epoch"] = [1]
            cdnv_history = {'cdnv': [cdnv_vals.mean()]}
            cdnv_history["epoch"] = [1]
        else:
            train_history['loss'].append(loss_record)
            train_history['acc'].append(acc_record)
            train_history["epoch"].append(i+1)
            cdnv_history['cdnv'].append(cdnv_vals.mean())
            cdnv_history["epoch"].append(i+1)

        if (i+1)%10 == 0:
            torch.save(params.model.state_dict(), (EXP_PATH / ("chkp_"+str(i+1))))

        if (i+1) % params.inspect_step == 0 or i==0:
            val_loss, val_acc = validation(params.model, criterion, check_dataloader, llm_mode=args.llm_mode)
            print("#####Val loss", round(val_loss, 4), "Val accuracy", round(val_acc, 4))
            if i == 0:
                test_history = {'loss': [val_loss], 'acc': [val_acc]}
                test_history["epoch"] = [1]
            else:
                test_history["epoch"].append(i+1)
                test_history['loss'].append(val_loss)
                test_history['acc'].append(val_acc)

    pickle.dump(test_history, open((EXP_PATH / "test_history.pkl"), "wb"))
    pickle.dump(train_history, open((EXP_PATH / "train_history.pkl"), "wb"))
    pickle.dump(cdnv_history, open((EXP_PATH / "cdnv_history.pkl"), "wb"))

    test_loss, test_acc = validation(params.model, criterion, params.data_loaders['test'], llm_mode=args.llm_mode)
    print("#####Test loss", round(test_loss, 4), "Test accuracy", round(test_acc, 4))
