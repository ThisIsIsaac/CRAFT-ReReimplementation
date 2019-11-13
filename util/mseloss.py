import numpy as np
import torch
import torch.nn as nn


class Maploss(nn.Module):
    # some details in: https://github.com/clovaai/CRAFT-pytorch/issues/18#issuecomment-513258344
    def __init__(self, use_gpu = True):

        super(Maploss,self).__init__()
        self.loss_fn = torch.nn.MSELoss(reduce=False, size_average=False)

    def cumulative_batch_loss(self, pre_loss, loss_label):
        """Caps the number of negative pixel losses depending on 3 times the number of positive pixels, or 500 when
        there is no positive pixel.

        positive pixel loss - MSE loss at a positive pixel (positive pixel is pixel with confidence >= 0.1)
        negative pixel loss - MSE loss at a negative pixel loss


        The ratio of positive-negative pixel losses is 3:1

        source: https://github.com/clovaai/CRAFT-pytorch/issues/18#issuecomment-513258344

        :param pre_loss:
        :param loss_label:
        :return:
        """

        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1))*0
        pre_loss = pre_loss.view(batch_size, -1) # [batch_size x 384 x 384] -> [batch_size x 147456]
        loss_label = loss_label.view(batch_size, -1) # [batch_size x 384 x 384] -> [batch_size x 147456]

        # iterate through all images of a single batch and accumulate the loss at sum_loss
        for i in range(batch_size):
            positive_pixel_mask = loss_label[i] >= 0.1
            negative_pixel_mask = loss_label[i] < 0.1

            num_positive_pixel_losses = len(pre_loss[i][positive_pixel_mask])
            num_negative_pixel_losses = len(pre_loss[i][negative_pixel_mask])

            if num_positive_pixel_losses != 0:
                posi_loss = torch.mean(pre_loss[i][positive_pixel_mask])
                sum_loss += posi_loss

                # cap the number of negative pixel losses to the largest (3*num_positive_pixel_losses) number of pixel losses
                if num_negative_pixel_losses < 3*num_positive_pixel_losses:
                    nega_loss = torch.mean(pre_loss[i][negative_pixel_mask])

                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][negative_pixel_mask], 3*num_positive_pixel_losses)[0])

                sum_loss += nega_loss

            # if there is no positive pixel, only add largest 500 negative pixel losses
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 500)[0])
                sum_loss += nega_loss

        return sum_loss

    def forward(self, gh_label, gah_label, p_gh, p_gah, mask):

        assert p_gh.size() == gh_label.size() and p_gah.size() == gah_label.size()
        loss1 = self.loss_fn(p_gh, gh_label)
        loss2 = self.loss_fn(p_gah, gah_label)
        loss_g = torch.mul(loss1, mask)
        loss_a = torch.mul(loss2, mask)

        # cumulative loss over the entire batch
        char_loss = self.cumulative_batch_loss(loss_g, gh_label)
        affi_loss = self.cumulative_batch_loss(loss_a, gah_label)

        return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]