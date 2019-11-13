import torch
from tqdm import tqdm
from util.mseloss import Maploss
import math
import logging

def validate(net, val_loader, writer, step):
    net.eval()
    loss_sum=0.0
    criterion = Maploss()
    num_val_images = len(val_loader.sampler)

    with torch.no_grad():
        tqdm_loader = tqdm(val_loader, desc="Validating")

        for (images, gh_label, gah_label, mask, _, img_paths) in tqdm_loader:
            images = images.cuda()
            gh_label = gh_label.cuda()
            gah_label = gah_label.cuda()
            mask = mask.cuda()

            out, _ = net(images)

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()

            loss = criterion(gh_label, gah_label, out1, out2, mask)
            loss_sum += loss.item()

            if loss > 1e8 or math.isnan(loss):
                imgs_paths_str = ""
                for img_path  in img_paths:
                    imgs_paths_str += img_path + "\n"

                logging.error("loss %.01f at validation, step %d!" % (loss, step))
                logging.error("above error occured while processing images:\n" + imgs_paths_str)
                writer.log_training(loss, step)

    net.train()
    avg_loss = loss_sum / num_val_images
    writer.log_validation(avg_loss, net, images.cpu(), out1.cpu(), out2.cpu(), step, save_image=False)