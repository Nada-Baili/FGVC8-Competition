import os, torch, time, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from focal_loss.focal_loss import FocalLoss

from utils import *
import prepare_data
from transform import *
from ResNext import resnext
from model import *

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--Pretrained_model', default="None", help="Path to the weights of the pretrained model")

def Train(pretrained_model):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    today = datetime.now()
    output_path = "./results/{}".format(today.strftime("%H_%M_%S_%d_%m_%Y"))
    os.mkdir(output_path)
    figures_path = os.path.join(output_path, "Plots")
    if not os.path.exists(figures_path):
        os.mkdir(figures_path)

    labels = pd.read_csv('./data/label_map.csv')
    train = pd.read_csv('./data/train-from-kaggle.csv')

    folds = train.copy()
    folds = make_folds(folds, params["n_folds"], params["SEED"])
    for FOLD in range(params["n_epochs"]):

        trn_idx = folds[folds['fold'] != FOLD].index
        val_idx = folds[folds['fold'] == FOLD].index

        train_dataset = prepare_data.Data(params,
                                          folds.loc[trn_idx].reset_index(drop=True),
                                          folds.loc[trn_idx]['attribute_ids'],
                                          transform=get_transforms(data='train'),
                                          is_Train=True)
        valid_dataset = prepare_data.Data(params,
                                          folds.loc[val_idx].reset_index(drop=True),
                                          folds.loc[val_idx]['attribute_ids'],
                                          transform=get_transforms(data='valid'),
                                          is_Train=True)

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=False)

        optimizer = Adam(model.parameters(), lr=params["lr"], amsgrad=False)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=4, verbose=True, eps=1e-6)
        #criterion = nn.BCEWithLogitsLoss(reduction='mean')
        criterion = FocalLoss(alpha=1, gamma=2, reduction="mean")
        #criterion = FocalLoss()

        best_score = 0.
        best_thresh = 0.
        best_loss = np.inf
        train_loss_epochs, val_loss_epochs = [], []

        for epoch in range(params["n_epochs"]):

            start_time = time.time()

            model.train()
            avg_loss = 0.

            optimizer.zero_grad()
            tk0 = tqdm(enumerate(train_loader), total=len(train_loader))

            for i, (images, labels) in tk0:
                images = images.to(device)
                labels = labels.to(device)
                y_preds = model(images)
                y_preds = torch.sigmoid(y_preds)
                loss = criterion(y_preds, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item() / len(train_loader)

            train_loss_epochs.append(avg_loss)

            model.eval()
            avg_val_loss = 0.
            preds = []
            valid_labels = []
            tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))

            for i, (images, labels) in tk1:
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    y_preds = model(images)

                preds.append(torch.sigmoid(y_preds).to('cpu').numpy())
                valid_labels.append(labels.to('cpu').numpy())

                loss = criterion(y_preds, labels)
                avg_val_loss += loss.item() / len(valid_loader)

            val_loss_epochs.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            preds = np.concatenate(preds)
            valid_labels = np.concatenate(valid_labels)
            argsorted = preds.argsort(axis=1)

            th_scores = {}
            for threshold in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]:
                _score = get_score(valid_labels, binarize_prediction(preds, threshold, argsorted))
                th_scores[threshold] = _score

            max_kv = max(th_scores.items(), key=lambda x: x[1])
            th, score = max_kv[0], max_kv[1]

            elapsed = time.time() - start_time

            if score > best_score:
                best_score = score
                best_thresh = th
                torch.save(model.state_dict(), os.path.join(output_path, "Fold{}_BestScore[{:4f}]_BestTh[{}].pth".format(FOLD+1, best_score, best_thresh)))

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(output_path, "Fold{}_BestLoss[{:4f}].pth".format(FOLD+1, best_loss)))

            plt.figure()
            plt.plot(train_loss_epochs, m="o", label="Train")
            plt.plot(val_loss_epochs, m="^", label="Val")
            plt.legend()
            if not os.path.exists(os.path.join(figures_path, "Fold{}".format(FOLD+1))):
                os.mkdir(os.path.join(figures_path, "Fold{}".format(FOLD+1)))
            plt.savefig(os.path.join(figures_path, "Fold{}/epoch{}.png".format(FOLD+1, epoch+1)))

            if epoch == params["Lr_decay_epoch"]:
                optimizer.param_groups[0]["lr"] *= params["Lr_decay"]
        torch.save(model.state_dict(), os.path.join(output_path, "Fold{}_LastModel.pth".format(FOLD + 1)))

if __name__ == '__main__':
    args = parser.parse_args()
    Train(args.Pretrained_model)