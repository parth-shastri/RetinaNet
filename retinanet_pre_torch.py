import math
import csv
from engine import train_one_epoch, evaluate
import torch
import transforms
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import config
from PIL import Image
import os
import utils

DATASET_PATH = "/content/drive/MyDrive/dsai_cv_dataset_combined/images"
ANNOTATIONS_PATH = "/content/drive/MyDrive/dsai_cv_dataset_combined/annotationsNormalized.csv"

n_classes = 2


def get_model(num_classes=2):
    retina_net = models.detection.retinanet_resnet50_fpn(pretrained=True)

    in_channels = retina_net.head.classification_head.conv[0].in_channels
    num_anchors = retina_net.head.classification_head.num_anchors
    retina_net.head.classification_head.num_classes = num_classes

    cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    nn.init.normal_(cls_logits.weight, std=0.01)
    nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    retina_net.head.classification_head.cls_logits = cls_logits

    print(in_channels, num_anchors)
    print(retina_net)
    return retina_net


# if torch.cuda.is_available():
#     retina_net.to(device)


class DSAIdata(Dataset):
    def __init__(self, data_csv, transforms=None):
        self.transforms = transforms
        self.data_csv = data_csv
        try:
            with open(self.data_csv, "r") as fp:
                read = csv.reader(fp)
                next(read)
                self.bboxes = []
                self.file_paths = []
                self.class_ids = []
                clas = "number_plate"

                for rows in read:
                    if rows[0].startswith("l"):
                        new_row = []
                        self.class_ids.append(config.CLASSES_CONFIG[clas])
                        self.file_paths.append(os.path.join(DATASET_PATH, rows[0]) if rows[0].endswith(".jpeg")
                                               else os.path.join(DATASET_PATH, rows[0] + ".jpeg"))
                        new_row.append(float(rows[3]) * float(rows[1]))
                        new_row.append(float(rows[4]) * float(rows[2]))
                        new_row.append(float(rows[5]) * float(rows[1]))
                        new_row.append(float(rows[6]) * float(rows[2]))
                        self.bboxes.append(new_row)
        except FileNotFoundError:
            print("The file deos not exist!")

    def __getitem__(self, item):
        img = Image.open(self.file_paths[item]).convert("RGB")
        bbox = self.bboxes[item]
        bbox = torch.as_tensor(bbox, dtype=torch.float32)
        bbox = torch.unsqueeze(bbox, dim=0)

        areas = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((bbox.shape[0],), dtype=torch.int64)

        target = dict()
        target["boxes"] = bbox
        target["labels"] = torch.tensor([self.class_ids[item]], dtype=torch.int64)
        target["image_id"] = torch.tensor([item])
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.file_paths)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    # These transforms are already handled by torch vision
    # transforms.Resize(size=800, max_size=1333),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # torchvision.transforms.Pad(padding=0),
])

test_transforms = transforms.Compose([transforms.ToTensor()])

dataset = DSAIdata(ANNOTATIONS_PATH, transforms=train_transforms)
test_dataset = DSAIdata(ANNOTATIONS_PATH, transforms=test_transforms)

indices = np.random.permutation(len(dataset))
train_dataset = torch.utils.data.Subset(dataset, indices[:-20])

print("Train Images found : ", len(train_dataset))
image, target = train_dataset[67]

print("Sample testing !!")
print(image.shape, "\n", target)

test_dataset = torch.utils.data.Subset(test_dataset, indices[-20:])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True,
                              collate_fn=utils.collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True,
                             collate_fn=utils.collate_fn)

images, targets = next(iter(train_dataloader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]

model = get_model()


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9,
                            weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, test_dataloader, device=device)


