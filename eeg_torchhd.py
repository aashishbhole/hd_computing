import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm
import copy

import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
from eeg import EEG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 10000  # number of hypervector dimensions
NUM_LEVELS = 100
BATCH_SIZE = 1
EPOCHS = 1
WINDOW_SIZE = 4


class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.id = embeddings.Random(size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

    def forward(self, x):
        sample_hv = torchhd.bind(self.id.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


train_dataset = EEG("bonn", "./data", train=True, window_size=WINDOW_SIZE)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ld = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = EEG("bonn", "./data", train=False, window_size=4)
test_ld = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

encode = Encoder(DIMENSIONS, train_ds[0][0].size(-1))
encode = encode.to(device)

num_classes = len(train_dataset.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)

best_model = None
best_val_acc = 0
best_val_acc_epoch_no = 0
for i in range(EPOCHS):
    with torch.no_grad():
        for samples, labels in tqdm(train_ld, desc="Training"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            if i == 0:
                print(samples_hv.shape)
                print(labels.shape)
                model.add(samples_hv, labels)
            else:
                model.add_online(samples_hv, labels)

    accuracy = torchmetrics.Accuracy("binary", num_classes=num_classes)
    with torch.no_grad():
        model.normalize()

        for samples, labels in tqdm(test_ld, desc="Validation"):
            samples = samples.to(device)
            labels = labels.to(device)

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            preds = outputs.argmax(1)
            accuracy.update(preds.cpu(), labels)
    print(f"\nValidation accuracy of {(accuracy.compute().item() * 100):.3f}%")
    if accuracy.compute().item() * 100 > best_val_acc:
        best_val_acc_epoch_no = i+1
        best_val_acc = accuracy.compute().item() * 100
        best_model = copy.deepcopy(model)
    else:
        print("Current best accuracy:", best_val_acc)


accuracy = torchmetrics.Accuracy("binary", num_classes=num_classes)
precision = torchmetrics.Precision("binary", num_classes=num_classes)
recall = torchmetrics.Recall("binary", num_classes=num_classes)
f1_score = torchmetrics.F1Score("binary", num_classes=num_classes)
specificity = torchmetrics.Specificity("binary", num_classes=num_classes)

with torch.no_grad():
    best_model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = best_model(samples_hv, dot=True)
        preds = outputs.argmax(1)
        accuracy.update(preds.cpu(), labels)
        precision.update(preds.cpu(), labels)
        recall.update(preds.cpu(), labels)
        f1_score.update(preds.cpu(), labels)
        specificity.update(preds.cpu(), labels)


print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
print(f"Testing precision of {(precision.compute().item() * 100):.3f}%")
print(f"Testing recall/sensitivity of {(recall.compute().item() * 100):.3f}%")
print(f"Testing F1 score of {(f1_score.compute().item() * 100):.3f}%")
print(f"Testing specificity of {(specificity.compute().item() * 100):.3f}%")