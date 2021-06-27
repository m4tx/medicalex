import logging
import sys

import torch.nn.functional as F
from inferno.io.box import Cityscapes
from inferno.io.box.cityscapes import CITYSCAPES_MEAN, CITYSCAPES_STD, CITYSCAPES_CLASSES_TO_LABELS
from inferno.io.core import Concatenate
from inferno.io.transform import Compose
from inferno.io.transform.generic import NormalizeRange, Normalize, Project, Cast, AsTorchBatch
from inferno.io.transform.image import RandomGammaCorrection, PILImage2NumPyArray, Scale, RandomFlip
from torch import optim, nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from datasets import *
from model.transunet import TransUNet
from pytorch_unet.dice_loss import dice_coeff
from pytorch_unet.train import dir_checkpoint

IMG_SIZE = (256, 256)


def make_transforms(image_shape, labels_as_onehot):
    # Make transforms
    image_transforms = Compose(PILImage2NumPyArray(),
                               NormalizeRange(),
                               RandomGammaCorrection(),
                               Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD))
    label_transforms = Compose(PILImage2NumPyArray(),
                               Project(projection=CITYSCAPES_CLASSES_TO_LABELS))
    joint_transforms = Compose(
        # RandomSizedCrop(ratio_between=(0.6, 1.0),
        #                 preserve_aspect_ratio=True),
        # # Scale raw image back to the original shape
        Scale(output_image_shape=image_shape,
              interpolation_order=3, apply_to=[0]),
        # Scale segmentation back to the original shape
        # (without interpolation)
        Scale(output_image_shape=image_shape,
              interpolation_order=0, apply_to=[1]),
        RandomFlip(allow_ud_flips=False),
        # Cast raw image to float
        Cast('float', apply_to=[0]))

    joint_transforms.add(Cast('long', apply_to=[1]))
    # Batchify
    joint_transforms.add(AsTorchBatch(2, add_channel_axis_if_necessary=False))
    # Return as kwargs
    return {'image_transform': image_transforms,
            'label_transform': label_transforms,
            'joint_transform': joint_transforms}


def get_cityscapes_loaders(root_directory, image_shape=(1024, 2048), labels_as_onehot=False,
                           include_coarse_dataset=False, read_from_zip_archive=True,
                           train_batch_size=1, validate_batch_size=1, num_workers=2):
    # Build datasets
    train_dataset = Cityscapes(root_directory, split='train',
                               read_from_zip_archive=read_from_zip_archive,
                               **make_transforms(image_shape, labels_as_onehot))
    if include_coarse_dataset:
        # Build coarse dataset
        coarse_dataset = Cityscapes(root_directory, split='train_extra',
                                    read_from_zip_archive=read_from_zip_archive,
                                    **make_transforms(image_shape, labels_as_onehot))
        # ... and concatenate with train_dataset
        train_dataset = Concatenate(coarse_dataset, train_dataset)
    validate_dataset = Cityscapes(root_directory, split='validate',
                                  read_from_zip_archive=read_from_zip_archive,
                                  **make_transforms(image_shape, labels_as_onehot))

    # Build loaders
    train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)
    validate_loader = data.DataLoader(validate_dataset, batch_size=validate_batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader, validate_loader


train_loader, validate_loader = get_cityscapes_loaders('./cities', IMG_SIZE, train_batch_size=1,
                                                       validate_batch_size=1, num_workers=10)

print('Train dataset size:', len(train_loader.dataset))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

net = TransUNet(n_classes=20, img_size=IMG_SIZE)
net.load_from(weights=np.load('./project_TransUNet/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'))
net = net.to(device=device)

summary(net, (1, 3, *IMG_SIZE), col_names=("input_size", "output_size", "num_params"),
        depth=4)


def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


def eval_net(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    los = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for x, y in loader:
            imgs, true_masks = x, y
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                los += F.cross_entropy(mask_pred, true_masks).item()
                tot += mIOU(true_masks, mask_pred)
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val, los / n_val


def train_net(net,
              device,
              train_loader,
              val_loader,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True):
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for x, y in train_loader:

                imgs = x.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = y.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (4 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score, val_loss = eval_net(net, val_loader, device)
                    scheduler.step(val_loss)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation IoU: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_loss, global_step)
                        writer.add_scalar('IoU/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


try:
    train_net(net=net,
              device=device,
              train_loader=train_loader,
              val_loader=validate_loader,
              epochs=5,
              batch_size=1,
              lr=0.003)
except KeyboardInterrupt:
    torch.save(net.state_dict(), 'INTERRUPTED.pth')
    logging.info('Saved interrupt')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
torch.save(net.state_dict(), 'MODEL.pth')
