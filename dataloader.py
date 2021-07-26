import csv
import numpy as np
import tensorflow as tf
import config
import torch

file_path = 'annotationsNormalized.csv'

with open(file_path, "r") as fp:
    read = csv.reader(fp)
    next(read)
    bboxes = []
    file_paths = []
    class_ids = []
    clas = "number_plate"

    for rows in read:
        new_row = []
        class_ids.append(config.CLASSES_CONFIG[clas])
        file_paths.append(rows[0])
        new_row.append(float(rows[3]))
        new_row.append(float(rows[4]))
        new_row.append(float(rows[5]))
        new_row.append(float(rows[6]))
        bboxes.append(new_row)

    print(bboxes)
    print(f"The total bounding boxes are : {len(bboxes)}")
    print(file_paths)
    print(class_ids)


file_paths = np.array(file_paths)

bboxes = np.array(bboxes, dtype=np.float32).reshape((len(bboxes), -1, 4))


class_ids = np.array(class_ids, dtype=np.uint8).reshape((len(bboxes), -1, 1))

dataset = tf.data.Dataset.from_tensor_slices((file_paths, bboxes, class_ids))

d = dataset.take(1)
print(d)
for i, b, c in d:
    print(i)
    print(b)
    print(c)

