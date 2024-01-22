import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

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
BATCH_SIZE = 10  # for GPUs with enough memory we can process multiple images at ones
EPOCHS = 100


class Encoder(nn.Module):
    def __init__(self, num_classes, size):
        super(Encoder, self).__init__()
        self.id = embeddings.Random(size, DIMENSIONS)
        self.value = embeddings.Level(NUM_LEVELS, DIMENSIONS)

    def forward(self, x):
        sample_hv = torchhd.bind(self.id.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)


class HDClustering(nn.Module):
    def __init__(self, num_clusters):
        super(HDClustering, self).__init__()
        self.centroids = Centroid(DIMENSIONS, num_clusters)
        self.centroids.weight = torch.nn.Parameter(torch.rand(num_clusters, DIMENSIONS))

    def fit(self, x):
        sim = torchhd.cosine_similarity(x, self.centroids.weight)
        clusters = sim.argmax(1)
        self.centroids.add(x, clusters)

    def forward(self, x):
        return self.centroids(x)


train_ds = EEG("bonn", "./data", train=True)
train_ld = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = EEG("bonn", "./data", train=False)
test_ld = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

encode = Encoder(DIMENSIONS, train_ds[0][0].size(-1))
encode = encode.to(device)

#KMeans
num_clusters = len(train_ds.classes)
model = KMeans(n_clusters=num_clusters)

with torch.no_grad():
    for samples, _ in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        model.fit(samples_hv.cpu())

accuracy = torchmetrics.Accuracy("binary", num_classes=num_clusters)
precision = torchmetrics.Precision("binary", num_classes=num_clusters)
recall = torchmetrics.Recall("binary", num_classes=num_clusters)
f1_score = torchmetrics.F1Score("binary", num_classes=num_clusters)
specificity = torchmetrics.Specificity("binary", num_classes=num_clusters)

with torch.no_grad():
    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        preds = torch.from_numpy(model.predict(samples_hv.cpu()))
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1_score.update(preds, labels)
        specificity.update(preds, labels)


print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
print(f"Testing precision of {(precision.compute().item() * 100):.3f}%")
print(f"Testing recall/sensitivity of {(recall.compute().item() * 100):.3f}%")
print(f"Testing F1 score of {(f1_score.compute().item() * 100):.3f}%")
print(f"Testing specificity of {(specificity.compute().item() * 100):.3f}%")


# #HD Clustering
# num_clusters = len(train_ds.classes)
# hd_model = HDClustering(num_clusters)
# hd_model = hd_model.to(device)
#
# with torch.no_grad():
#     for samples, _ in tqdm(train_ld, desc="Training"):
#         samples = samples.to(device)
#
#         samples_hv = encode(samples)
#         hd_model.fit(samples_hv)
#
# accuracy = torchmetrics.Accuracy("binary", num_classes=num_clusters)
# precision = torchmetrics.Precision("binary", num_classes=num_clusters)
# recall = torchmetrics.Recall("binary", num_classes=num_clusters)
# f1_score = torchmetrics.F1Score("binary", num_classes=num_clusters)
# specificity = torchmetrics.Specificity("binary", num_classes=num_clusters)
#
# with torch.no_grad():
#     for samples, labels in tqdm(test_ld, desc="Testing"):
#         samples = samples.to(device)
#
#         samples_hv = encode(samples)
#         preds = hd_model(samples_hv).cpu().argmax(1)
#         accuracy.update(preds, labels)
#         precision.update(preds, labels)
#         recall.update(preds, labels)
#         f1_score.update(preds, labels)
#         specificity.update(preds, labels)
#
#
# print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
# print(f"Testing precision of {(precision.compute().item() * 100):.3f}%")
# print(f"Testing recall/sensitivity of {(recall.compute().item() * 100):.3f}%")
# print(f"Testing F1 score of {(f1_score.compute().item() * 100):.3f}%")
# print(f"Testing specificity of {(specificity.compute().item() * 100):.3f}%")
