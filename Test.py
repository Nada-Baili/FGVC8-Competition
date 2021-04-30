import os, torch, time, argparse
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader
from torchsummary import summary
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

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
parser.add_argument('--Weights_Path', default="./results/Fold1_BestScore[0.566719]_BestTh[0.05].pth", help="Path to the weights of the trained model")

def Test(weights_path, min_labels=1, max_labels=5):
    if not os.path.exists("./submissions"):
        os.mkdir("./submissions")

    submission = pd.read_csv('./data/sample_solution.csv')
    submission = submission.iloc[:5]
    test_dataset = prepare_data.Data(params,
                                      submission,
                                      transform=get_transforms(data='valid'),
                                      is_Train=False)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    model = se_resnext50_32x4d("./pretrained_models/se_resnext50_32x4d.pth").to(device)
    model.load_state_dict(torch.load(weights_path))

    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, images in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        preds.append(torch.sigmoid(y_preds).to('cpu').numpy())
    preds = np.concatenate(preds)
    sorted_preds = np.sort(preds)

    threshold = float(os.path.split(weights_path)[1].split("_")[-1][7:-5])
    predictions = sorted_preds > threshold
    for i, row in enumerate(predictions):
        ids = np.nonzero(row)[0]
        ids = ids[-max(min_labels, min(len(ids), max_labels)):]
        submission.iloc[i].attribute_ids = ' '.join([str(x) for x in ids])

    today = datetime.now()
    output_path = "./submissions/{}".format(today.strftime("%H_%M_%S_%d_%m_%Y"))
    os.mkdir(output_path)
    submission.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
    submission.head()

if __name__ == '__main__':
    args = parser.parse_args()
    Test(args.Weights_Path)