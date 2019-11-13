# source: https://github.com/seungwonpark/melgan/blob/master/utils/writer.py
from tensorboardX import SummaryWriter
import logging

class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)
        self.is_first = True

    def log_training(self, loss, step):
        self.add_scalar('train.loss', loss, step)

    def log_validation(self, loss, net, input_image, region_score, affinity_score, step, save_image=False):
        logging.info("step = " + str(step) + ", loss = " + str(loss))

        self.add_scalar('validation.loss', loss, step)
        self.log_histogram(net, step)

        if save_image:
            self.add_image("input_image", input_image, step)
            self.add_image("region_score", region_score, step)
            self.add_image("affinity_score", affinity_score, step)

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)