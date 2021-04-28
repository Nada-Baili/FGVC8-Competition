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

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--Weights_Path', default="./model_paths_1000.pth", help="Path to the weights of the trained model")

def Test(weights_path):
    submission = pd.read_csv('./data/sample_solution.csv')
    test_dataset = TestDataset(submission, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = resnext(params["nb_classes"]).to(device)
    model.load_state_dict(torch.load(weights_path))

    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, images in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        preds.append(torch.sigmoid(y_preds).to('cpu').numpy())

    threshold = best_thresh
    predictions = np.concatenate(preds) > threshold
    for i, row in enumerate(predictions):
        ids = np.nonzero(row)[0]
        submission.iloc[i].attribute_ids = ' '.join([str(x) for x in ids])

    today = datetime.now()
    output_path = "./submissions/{}".format(today.strftime("%H_%M_%S_%d_%m_%Y"))
    os.mkdir(output_path)
    submission.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
    submission.head()

if __name__ == '__main__':
    args = parser.parse_args()
    Test()