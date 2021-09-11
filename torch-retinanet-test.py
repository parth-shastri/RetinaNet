import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from torchvision.transforms.functional import convert_image_dtype
import torchvision.transforms.functional as F


from retinanet_pre_torch import get_model

weights_path = "torchvision-retinanet/tv_retinanet_romanian.pth"
img_path = "car_rear.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inf_model = get_model(2)

state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

inf_model.load_state_dict(state_dict)

test_image = Image.open(img_path).convert("RGB")
test_image = np.asarray(test_image)

transformed_image = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])(test_image)

inf_model.eval()

with torch.no_grad():
    pred = inf_model([transformed_image])

#%%

print(pred)
#%%


def show_pred(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    figs, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):

        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
    plt.show()


thresh = 0.25

image = convert_image_dtype(transformed_image, dtype=torch.uint8)

images = draw_bounding_boxes(image, boxes=pred[0]["boxes"][pred[0]["scores"] > thresh], width=4)

show_pred(images)

