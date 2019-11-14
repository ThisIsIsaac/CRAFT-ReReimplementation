# source: https://github.com/seungwonpark/melgan/blob/master/utils/writer.py
from tensorboardX import SummaryWriter
import logging

class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)
        self.is_first = True

    def log_training(self, loss, step):
        self.add_scalar('train.loss', loss, step)

    def log_validation(self, loss, net, step, my_outputs=None, ref_outputs=None, save_image=False):
        logging.info("step = " + str(step) + ", loss = " + str(loss))

        self.add_scalar('validation.loss', loss, step)
        self.log_histogram(net, step)

        if save_image:
            self.log_output_images(my_outputs, ref_outputs)

    def log_output_images(self, my_outputs, ref_outputs, step):
        """Inputs must be the returned list of images from `craft_util.save_outputs_from_tensors`.

        :param my_outputs:
        :param ref_outputs:
        :param step:
        :return:
        """

        batch_size = len(my_outputs)
        assert batch_size == len(ref_outputs)

        for i in range(batch_size):
            self.add_image("output_image", my_outputs[i], step, dataformats="HWC")
            self.add_image("ref_image", ref_outputs[i], step, dataformats="HWC")

    def log_histogram(self, model, step):
        for tag, value in model.named_parameters():
            self.add_histogram(tag.replace('.', '/'), value.cpu().detach().numpy(), step)