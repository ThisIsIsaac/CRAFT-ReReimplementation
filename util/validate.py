import torch
from tqdm import tqdm
from util.mseloss import Maploss
import math
import logging
from util import craft_utils
import os

def validate(args, net, val_loader, writer, step, images_path):
    net.eval()
    loss_sum=0.0
    criterion = Maploss()
    num_val_images = len(val_loader.sampler)

    is_first_iteration = True

    with torch.no_grad():
        tqdm_loader = tqdm(val_loader, desc="Validating")

        for (images, gh_label, gah_label, mask, _, unnormalized_images, img_paths) in tqdm_loader:
            images = images.cuda()
            gh_label = gh_label.cuda()
            gah_label = gah_label.cuda()
            mask = mask.cuda()

            out, _ = net(images)

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()

            loss = criterion(gh_label, gah_label, out1, out2, mask)
            loss_sum += loss.item()

            # if log explodes, save images and log approriate data
            if loss > 1e8 or math.isnan(loss):
                imgs_paths_str = ""
                for img_path  in img_paths:
                    imgs_paths_str += img_path + "\n"

                # log error message
                logging.error("loss %.01f during validation, at step %d!" % (loss, step))
                logging.error("above error occured while processing images:\n" + imgs_paths_str)

                # create path and directories to save imags
                error_images_path = os.path.join(images_path, "valid_error")
                if not os.path.exists(error_images_path):
                    os.mkdir(error_images_path)

                output_images_path = os.path.join(error_images_path, "network_" + str(step))
                if not os.path.exists(output_images_path):
                    os.mkdir(output_images_path)

                ref_images_path = os.path.join(error_images_path, "ref_" + str(step))
                if not os.path.exists(ref_images_path):
                    os.mkdir(ref_images_path)

                # save images to disk
                output_images = craft_utils.save_outputs_from_tensors(unnormalized_images, out1, out2,
                                                                      args.text_threshold, args.link_threshold,
                                                                      args.low_text,
                                                                      output_images_path, img_paths)
                ref_images = craft_utils.save_outputs_from_tensors(unnormalized_images, gh_label, gah_label,
                                                                   args.text_threshold, args.link_threshold,
                                                                   args.low_text,
                                                                   ref_images_path, img_paths)

                # log loss and images
                writer.log_validation(loss, net, step, my_outputs=output_images, ref_outputs=ref_images, save_images=True)

            # save the first batch of images
            if is_first_iteration:
                # create path and directories to save imags
                valid_images_path = os.path.join(images_path, "valid")
                if not os.path.exists(valid_images_path):
                    os.mkdir(valid_images_path)

                output_images_path = os.path.join(valid_images_path, "network_" + str(step))
                if not os.path.exists(output_images_path):
                    os.mkdir(output_images_path)

                ref_images_path = os.path.join(valid_images_path, "ref_" + str(step))
                if not os.path.exists(ref_images_path):
                    os.mkdir(ref_images_path)

                output_images = craft_utils.save_outputs_from_tensors(unnormalized_images, out1, out2,
                                                                      args.text_threshold, args.link_threshold,
                                                                      args.low_text,
                                                                      output_images_path, img_paths)
                ref_images = craft_utils.save_outputs_from_tensors(unnormalized_images, gh_label, gah_label,
                                                                   args.text_threshold, args.link_threshold,
                                                                   args.low_text,
                                                                   ref_images_path, img_paths)

                writer.log_validation(loss, net, step, my_outputs=output_images, ref_outputs=ref_images,
                                      save_images=True)
                is_first_iteration = False


    net.train()
    avg_loss = loss_sum / num_val_images
    writer.log_validation(avg_loss, net, step, save_images=False)
