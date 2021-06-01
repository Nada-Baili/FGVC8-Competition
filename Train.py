import os, torch, argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from focal_loss.focal_loss import FocalLoss
import torchvision.models as models

from utils import *
import prepare_data
from transform import *
from model import *

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--Model_Name', default="se_resnext50", help="se_resnext50, se_resnext101, resnet18, vgg16bn")

def Train(model_name):
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

    attributes = [l.split("::")[0] for l in list(labels["attribute_name"])]
    names = np.unique(attributes)

    folds = train.copy()
    folds = make_folds(folds, params["n_folds"], params["SEED"])
    for FOLD in range(params["n_folds"]):
        print("Fold {}".format(FOLD+1))
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

        if model_name == "se_resnext50":
            model = se_resnext50_32x4d().to(device)
        elif model_name == "se_resnext101":
            model = se_resnext101_32x4d().to(device)
        elif model_name == "resnet18":
            model=models.resnet18(pretrained=True)
            model.avgpool=nn.AdaptiveAvgPool2d(1)
            model.fc=nn.Linear(model.fc.in_features, params["nb_classes"])
            model=model.to(device)
        elif model_name == "vgg16bn":
            model=models.vgg16_bn(pretrained=True)
            model.classifier[6] = nn.Linear(4096,params["nb_classes"])
            model=model.to(device)

        optimizer = Adam(model.parameters(), lr=params["lr"])
        scheduler = CosineAnnealingLR(optimizer, params["n_epochs"], eta_min=0, verbose=True)
        criterion = FocalLoss(alpha=0.25, gamma=2, reduction="mean")

        best_score = 0.
        train_loss_epochs, val_loss_epochs = [], []

        for epoch in range(params["n_epochs"]):

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
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
                y_preds = torch.sigmoid(y_preds)
                preds.append(y_preds.to('cpu').numpy())
                valid_labels.append(labels.to('cpu').numpy())

                loss = criterion(y_preds, labels)
                avg_val_loss += loss.item() / len(valid_loader)

            val_loss_epochs.append(avg_val_loss)
            scheduler.step()

            preds = np.concatenate(preds)
            valid_labels = np.concatenate(valid_labels)
            argsorted = preds.argsort(axis=1)

            th_scores = {}
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,1.6,1.7,1.8,1.9,2.0]:
               binarized_labels = binarize_prediction(preds, threshold, argsorted, max_labels=5)
               _score = get_score(valid_labels, binarized_labels)
               th_scores[threshold] = _score

            max_kv = max(th_scores.items(), key=lambda x: x[1])
            th, score = max_kv[0], max_kv[1]

            if score > best_score:
                best_score = score
                best_thresh = th
                torch.save(model.state_dict(), os.path.join(output_path, "Fold{}_BestScore[{:4f}]_BestTh[{:4f}].pth".format(FOLD+1, best_score, best_thresh)))

            plt.figure()
            plt.plot(train_loss_epochs, marker="o", label="Train")
            plt.plot(val_loss_epochs, marker="^", label="Val")
            plt.legend()
            if not os.path.exists(os.path.join(figures_path, "Fold{}".format(FOLD+1))):
                os.mkdir(os.path.join(figures_path, "Fold{}".format(FOLD+1)))
            plt.savefig(os.path.join(figures_path, "Fold{}/epoch{}.png".format(FOLD+1, epoch+1)))

        torch.save(model.state_dict(), os.path.join(output_path, "Fold{}_LastModel.pth".format(FOLD + 1)))

if __name__ == '__main__':
    args = parser.parse_args()
    Train(args.Model_Name)
