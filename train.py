import json
import torch
from huggingface_hub import hf_hub_download
from torchvision.transforms import ColorJitter
from torch import nn
import evaluate
from torch.utils import data as torchdata

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from datasets import load_dataset, DatasetDict

import flor
from flor import MTK as Flor
import numpy as np 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_dataset_identifier = "segments/sidewalk-semantic"
ds = load_dataset(hf_dataset_identifier)
assert isinstance(ds, DatasetDict)

ds = ds.shuffle()
ds = ds["train"].train_test_split(test_size=0.1)
train_ds = ds["train"]
test_ds = ds["test"]

id2label = json.load(
    open(
        hf_hub_download(
            repo_id=hf_dataset_identifier, filename="id2label.json", repo_type="dataset"
        ),
        "r",
    )
)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", id2label=id2label, label2id=label2id
)
assert isinstance(feature_extractor, SegformerFeatureExtractor)
assert isinstance(model, SegformerForSemanticSegmentation)
model = model.to(device)
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)


def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels, return_tensors="pt")
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = feature_extractor(images, labels, return_tensors="pt")
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

metric = evaluate.load("mean_iou")


def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=feature_extractor.do_reduce_labels,
        )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update(
            {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
        )
        metrics.update(
            {f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)}
        )

        return metrics


epochs = flor.arg("epochs", 5)
lr = flor.arg("lr", 6e-5)
batch_size = flor.arg("batch_size", 4)


train_loader = torchdata.DataLoader(
    dataset=train_ds.with_transform(train_transforms),  # type: ignore
    batch_size=batch_size,
    shuffle=True,
    collate_fn=torchdata.default_collate,
)
test_loader = torchdata.DataLoader(
    dataset=test_ds.with_transform(val_transforms),  # type: ignore
    batch_size=batch_size,
    shuffle=False,
    collate_fn=torchdata.default_collate,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

Flor.checkpoints(model, optimizer)
total_steps = len(train_loader)
for epoch in Flor.loop(range(epochs)):
    model.train()
    for i, batch in Flor.loop(enumerate(train_loader)):
        for k in batch:
            v = batch[k]
            batch[k] = v.to(device) if hasattr(v, "to") else v
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, epochs, i + 1, total_steps, flor.log("loss", loss.item())
            )
        )

        if i == 49:  # flor.arg("num_steps", 49):
            break

print("Model TEST")

model.eval()
with torch.no_grad(): 
    for i, batch in enumerate(test_loader):
        for k in batch:
            v = batch[k]
            batch[k] = v.to(device) if hasattr(v, "to") else v
        outputs = model(**batch)
        labels = np.array(batch['labels'].cpu())
        logits = np.array(outputs.cpu().logits) 
        metrics = compute_metrics((logits, labels))
        print("mean_iou: ", metrics['mean_iou'])
