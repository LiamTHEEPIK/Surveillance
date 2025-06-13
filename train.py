import torch
import toml
import tqdm
import utils

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from models import AnomalyDetectionModel
from dataset import AnomalyDetectionDataset


def set_random_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pad_sequences(batch):
    batch_lengths = torch.stack([item[2] for item in batch])

    batch_inputs = [item[0] for item in batch]
    batch_labels = [item[1] for item in batch]

    batch_inputs = nn.utils.rnn.pad_sequence(batch_inputs, batch_first=True)
    batch_labels = nn.utils.rnn.pad_sequence(batch_labels, batch_first=True)

    return batch_inputs, batch_labels, batch_lengths


def dice_weight(probability, delta=0.1):
    return (probability / delta).clamp(min=0.0, max=1.0)


def bidirectional_dice_loss(outputs, targets):
    return 1 - utils.bidirectional_dice_score(outputs, targets, dice_weight(targets.sum() / targets.numel()))


def criterion(outputs, targets, lengths):
    batch_loss = torch.tensor(0).to(lengths.device).float()

    for batch, length in enumerate(lengths):
        output = outputs[batch, :length]
        target = targets[batch, :length]

        batch_loss += nn.functional.binary_cross_entropy(output, target) + bidirectional_dice_loss(output, target)

    return batch_loss / lengths.shape[0]


configs = toml.load('configs/config.toml')

train_dataset = AnomalyDetectionDataset('datasets/train')
valid_dataset = AnomalyDetectionDataset('datasets/valid')

train_dataset_size = len(train_dataset)
valid_dataset_size = len(valid_dataset)

set_random_seed(configs['seed'])

train_dataloader = DataLoader(train_dataset, batch_size=configs['batch-size'], num_workers=configs['num-workers'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=configs['group-size'], num_workers=configs['num-workers'], shuffle=False)

train_dataloader.collate_fn = pad_sequences
valid_dataloader.collate_fn = pad_sequences

train_dataloader_size = len(train_dataloader)
valid_dataloader_size = len(valid_dataloader)

best_iou_score = 0.0
last_iou_score = 0.0

device = torch.device(configs['device'])

attention_window = configs['attention-window']
smoothing_window = configs['smoothing-window']

num_epochs = configs['num-epochs']

model = AnomalyDetectionModel(attention_window, alpha=configs['alpha'])
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=configs['learning-rate'], weight_decay=configs['weight-decay'])

load_checkpoint_path = configs['load-checkpoint-path']
best_checkpoint_path = configs['best-checkpoint-path']
last_checkpoint_path = configs['last-checkpoint-path']

if configs['load-checkpoint']:
    model.load_state_dict(torch.load(load_checkpoint_path, map_location=device, weights_only=True))

print(f'\n---------- training start at: {device} ----------\n')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for inputs, labels, lengths in tqdm.tqdm(train_dataloader, ncols=80):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.sigmoid(), labels, lengths.to(device))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    train_loss /= train_dataloader_size

    with torch.no_grad():
        all_scores = torch.zeros(0).to(device)
        all_labels = torch.zeros(0).to(device)

        for inputs, labels, _ in tqdm.tqdm(valid_dataloader, ncols=80):
            inputs = inputs.to(device)
            labels = labels.to(device)

            scores = model(inputs).sigmoid()
            scores = utils.score_smoothing(scores, smoothing_window).repeat_interleave(16, dim=1)

            scores = scores.mean(dim=0)
            labels = labels.mean(dim=0)

            all_scores = torch.cat([all_scores, scores])
            all_labels = torch.cat([all_labels, labels])

        all_scores = all_scores.cpu()
        all_labels = all_labels.cpu()

        iou_score = utils.iou_score((all_scores > 0.5).int(), all_labels).item()

        if iou_score > best_iou_score:
            best_iou_score = iou_score
            torch.save(model.state_dict(), best_checkpoint_path)

        last_iou_score = iou_score
        torch.save(model.state_dict(), last_checkpoint_path)

    print(f'\nepoch: {epoch + 1}/{num_epochs:<6} loss: {train_loss:<10.5f} IoU: {iou_score:.3f}\n')

print(f'best IoU: {best_iou_score:.3f}')
print(f'last IoU: {last_iou_score:.3f}')

print('\n---------- training finished ----------\n')
