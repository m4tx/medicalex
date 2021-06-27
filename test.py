import logging

import matplotlib.pyplot as plt

from datasets import *
from model.transunet import TransUNet
from unet import UNet

test_dataset = make_h5_directory_dataset("project_TransUNet/data/Synapse/test_vol_h5")
print("Test dataset size:", len(test_dataset))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
logging.info(f"Using device {device}")

net = TransUNet(out_classes=10, img_size=(512, 512))
# logging.info(
#     f"Network:\n"
#     f"\t{net.input_channels} input channels\n"
#     f"\t{net.out_classes} output channels (classes)\n"
#     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling'
# )

net.to(device=device)
# faster convolutions, but more memory
# cudnn.benchmark = True
# model_path = "/home/sharley/Projects/medicalex/checkpoints/CP_epoch5.pth"
model_path = "MODEL.pth"
net.load_state_dict(torch.load(model_path))

losses = [list() for _ in range(8)]
for cnt, data in enumerate(test_dataset):
    if cnt % 10 == 0:
        print(f"{cnt} / {len(test_dataset)}")

    target = data["label"]
    target = target.view(-1)
    pred = net(data["image"].unsqueeze(0).to(device=device, dtype=torch.float32))
    pred = torch.argmax(pred, dim=1).view(-1)
    plt.imshow(pred.cpu().squeeze())
    plt.show()

    for current_class in range(1, 9):
        target_ind = (target == current_class).nonzero(as_tuple=True)[0]
        pred_ind = (pred == current_class).nonzero(as_tuple=True)[0]

        intersecting = len(np.intersect1d(target_ind.cpu(), pred_ind.cpu()))
        if len(target_ind) == 0 and len(pred_ind) == 0:
            continue
        dice_loss = (2 * intersecting) / (len(target_ind) + len(pred_ind))
        losses[current_class - 1].append(dice_loss)

for i, arr in enumerate(losses):
    print(f"Class {i}: {np.mean(arr)}")
