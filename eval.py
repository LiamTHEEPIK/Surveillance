import torch
import toml
import tqdm
import utils
import sklearn.metrics as metrics

from torch.utils.data import DataLoader

from models import AnomalyDetectionModel
from dataset import AnomalyDetectionDataset


def f1_score(scores, labels, threshold):
    return metrics.f1_score(labels, (scores > threshold).int())


def far_score(scores, labels, threshold):
    return utils.far_score(scores, labels, threshold)


def iou_score(scores, labels, threshold):
    return utils.iou_score((scores > threshold).int(), labels)


configs = toml.load('configs/config.toml')

dataset = AnomalyDetectionDataset('datasets/valid')
dataset_size = len(dataset)

dataloader = DataLoader(dataset, batch_size=configs['group-size'], num_workers=configs['num-workers'], shuffle=False)
dataloader_size = len(dataloader)

device = torch.device(configs['device'])

attention_window = configs['attention-window']
smoothing_window = configs['smoothing-window']

model = AnomalyDetectionModel(attention_window, alpha=configs['alpha'])
model = model.to(device)

print(f'\n---------- evaluation start at: {device} ----------\n')

with torch.no_grad():
    scores0 = torch.zeros(0).to(device)
    scores1 = torch.zeros(0).to(device)

    labels0 = torch.zeros(0).to(device)
    labels1 = torch.zeros(0).to(device)

    model.load_state_dict(torch.load(configs['load-checkpoint-path'], map_location=device, weights_only=True))
    model.eval()

    for inputs, labels, _ in tqdm.tqdm(dataloader, ncols=80):
        inputs = inputs.to(device)
        labels = labels.to(device)

        scores = model(inputs).sigmoid()
        scores = utils.score_smoothing(scores, smoothing_window).repeat_interleave(16, dim=1)

        scores = scores.mean(dim=0)
        labels = labels.mean(dim=0)

        if labels.sum() == 0:
            scores0 = torch.cat([scores0, scores])
            labels0 = torch.cat([labels0, labels])
        else:
            scores1 = torch.cat([scores1, scores])
            labels1 = torch.cat([labels1, labels])

    scores0 = scores0.cpu()
    scores1 = scores1.cpu()

    labels0 = labels0.cpu()
    labels1 = labels1.cpu()

    scores = torch.cat([scores0, scores1])
    labels = torch.cat([labels0, labels1])

    auc_score = metrics.roc_auc_score(labels, scores)

    f1_score20 = f1_score(scores, labels, 0.2)
    f1_score30 = f1_score(scores, labels, 0.3)
    f1_score40 = f1_score(scores, labels, 0.4)
    f1_score50 = f1_score(scores, labels, 0.5)

    ap_score = metrics.average_precision_score(labels, scores)

    far20 = far_score(scores0, labels0, 0.2)
    far30 = far_score(scores0, labels0, 0.3)
    far40 = far_score(scores0, labels0, 0.4)
    far50 = far_score(scores0, labels0, 0.5)

    iou20 = iou_score(scores1, labels1, 0.2)
    iou30 = iou_score(scores1, labels1, 0.3)
    iou40 = iou_score(scores1, labels1, 0.4)
    iou50 = iou_score(scores1, labels1, 0.5)

    print('\n--------------------------------')
    print(f'F1-Score@20: {f1_score20:.4f}')
    print(f'F1-Score@30: {f1_score30:.4f}')
    print(f'F1-Score@40: {f1_score40:.4f}')
    print(f'F1-Score@50: {f1_score50:.4f}')

    print('\n--------------------------------')
    print(f'FAR@20: {far20:.4f}')
    print(f'FAR@30: {far30:.4f}')
    print(f'FAR@40: {far40:.4f}')
    print(f'FAR@50: {far50:.4f}')

    print('\n--------------------------------')
    print(f'IoU@20: {iou20:.4f}')
    print(f'IoU@30: {iou30:.4f}')
    print(f'IoU@40: {iou40:.4f}')
    print(f'IoU@50: {iou50:.4f}')

    print(f'\nAUC: {auc_score:<8.4f} AP: {ap_score:.4f}')

print(f'\n---------- evaluation finished ----------\n')
