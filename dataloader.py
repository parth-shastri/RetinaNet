import torchvision.transforms
from PIL import Image
import config
import torch
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as et
import utils
import numpy as np
# file_path = 'annotationsNormalized.csv'
#
# with open(file_path, "r") as fp:
#     read = csv.reader(fp)
#     next(read)
#     bboxes = []
#     file_paths = []
#     class_ids = []
#     clas = "number_plate"
#
#     for rows in read:
#         new_row = []
#         class_ids.append(config.CLASSES_CONFIG[clas])
#         file_paths.append(rows[0])
#         new_row.append(float(rows[3]))
#         new_row.append(float(rows[4]))
#         new_row.append(float(rows[5]))
#         new_row.append(float(rows[6]))
#         bboxes.append(new_row)
#
#     print(bboxes)
#     print(f"The total bounding boxes are : {len(bboxes)}")
#     print(file_paths)
#     print(class_ids)
#
#
# file_paths = np.array(file_paths)
#
# bboxes = np.array(bboxes, dtype=np.float32).reshape((len(bboxes), -1, 4))
#
#
# class_ids = np.array(class_ids, dtype=np.uint8).reshape((len(bboxes), -1, 1))
#
# dataset = tf.data.Dataset.from_tensor_slices((file_paths, bboxes, class_ids))
#
# d = dataset.take(1)
# print(d)
# for i, b, c in d:
#     print(i)
#     print(b)
#     print(c)

def collate_fn(batch):
    return tuple(zip(*batch))


train_anno_path = r"C:\Users\shast\datasets\license-plate-dataset-master\dataset\train\annots"
train_path = r"C:\Users\shast\datasets\license-plate-dataset-master\dataset\train\images"


class RomanianTrafficData(Dataset):
    def __init__(self, data_dir, annotations_dir, transforms=None):
        self.data_dir = data_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms

    def __getitem__(self, item):
        filename = os.listdir(self.annotations_dir)[item]
        # Copy code for Pascal VOC style Annotations files
        tree = et.parse(os.path.join(self.annotations_dir, filename))

        root = tree.getroot()
        boxes = []
        class_ids = []

        img_path = os.path.join(self.data_dir, root.find("filename").text)
        img = Image.open(img_path).convert("RGB")

        for object in root.iter("object"):
            box = object.find("bndbox")
            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            class_id = config.CLASSES_CONFIG[object.find("name").text]
            class_ids.append(class_id)

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(class_ids, dtype=torch.int64)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0], ), dtype=torch.uint8)
        target = dict()

        target["boxes"] = boxes
        target["labels"] = labels
        target["iscrowd"] = iscrowd
        target["area"] = areas
        target["image_id"] = torch.tensor([item])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(os.listdir(self.data_dir))


train_dataset = RomanianTrafficData(train_path, train_anno_path)
print(len(train_dataset))
img, target = train_dataset[3]
print(f"The shape of the image {np.array(img).shape}\nThe target in COCO format {target}")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=utils.collate_fn)

print(len(train_dataloader))

