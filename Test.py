import os, torch, argparse
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import pandas as pd
from torch.utils.data import DataLoader

import prepare_data
from transform import *
from model import *
import torchvision.models as models

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--Weights_Path', default="./results/Fold1_BestScore[0.617404]_BestTh[0.100000].pth", help="Path to the weights of the trained model")
parser.add_argument('--Model_Name', default="se_resnext50", help="se_resnext50, se_resnext101, resnet18, vgg16bn")

def Test(weights_path, model_name, min_labels=1, max_labels=5):
    if not os.path.exists("./submissions"):
        os.mkdir("./submissions")

    submission = pd.read_csv('./data/sample_solution.csv')
    submission_ = submission.drop(143)
    idxs = submission_.index
    test_dataset = prepare_data.Data(params,
                                      submission_,
                                      transform=get_transforms(data='valid'),
                                      is_Train=False)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    if model_name == "se_resnext50":
        model = se_resnext50_32x4d().to(device)
    elif model_name == "se_resnext101":
        model = se_resnext101_32x4d().to(device)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, params["nb_classes"])
        model = model.to(device)
    elif model_name == "vgg16bn":
        model = models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096,params["nb_classes"])
        model = model.to(device)
    model.load_state_dict(torch.load(weights_path))

    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    model.eval()
    for i, images in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        y_preds = torch.sigmoid(y_preds)
        preds.append(y_preds.to('cpu').numpy())
    preds = np.concatenate(preds)

    sorted_preds = np.sort(preds, axis=1)
    sorted_preds_idxs = np.argsort(preds, axis=1)

    threshold = float(os.path.split(weights_path)[1].split("_")[-1][7:-5])


    predictions = sorted_preds > threshold
    for i, row in enumerate(predictions):
        ids = np.nonzero(row)[0]
        ids = ids[-max(min_labels, min(len(ids), max_labels)):]
        submission.iloc[idxs[i]].attribute_ids = ' '.join([str(sorted_preds_idxs[i, x]) for x in ids])
    submission.iloc[143].attribute_ids = '2369'

    today = datetime.now()
    output_path = "./submissions/{}".format(today.strftime("%H_%M_%S_%d_%m_%Y"))
    os.mkdir(output_path)
    submission.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
    submission.head()

if __name__ == '__main__':
    args = parser.parse_args()
    Test(args.Weights_Path, args.Model_Name)