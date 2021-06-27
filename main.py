import logging
import sys

from torchinfo import summary

from datasets import *
from model.transunet import TransUNet
from pytorch_unet.train import train_net

test_dataset = make_h5_directory_dataset("project_TransUNet/data/Synapse/test_vol_h5")
print("Test dataset size:", len(test_dataset))
print(test_dataset)

train_dataset = NpzDirectoryDataset("project_TransUNet/data/Synapse/train_npz")
print("Train dataset size:", len(train_dataset))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device {device}")


logging.info("Creating model")
net = TransUNet(out_classes=10, img_size=(512,512))
logging.info("Summary")
summary(
    net,
    [1] + list(train_dataset[0]["image"].shape), col_names=("input_size", "output_size", "num_params"), depth=4,
)

net.to(device=device)

try:
    train_net(
        net=net,
        device=device,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=350,
        batch_size=1,
        lr=0.01,
        img_scale=0.5,
    )
except KeyboardInterrupt:
    torch.save(net.state_dict(), "INTERRUPTED.pth")
    logging.info("Saved interrupt")
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
torch.save(net.state_dict(), "MODEL.pth")
