from data_loader.craft_base_dataset import craft_base_dataset
from util.file_utils import *
from util.mep import mep
import numpy as np
import cv2
import math


class SynthKorean(craft_base_dataset):
    def __init__(self, net, dataset_path, target_size=768, viz=False, debug=False, data_corruption=False):
        super(SynthKorean, self).__init__(target_size, viz, debug, perform_input_data_corruption=data_corruption)
        self.dataset_path = dataset_path
        self.net = net
        self.img_paths = []
        self.gt_paths = []
        self.viz = viz

        if not os.path.exists(dataset_path):
            raise FileNotFoundError("Given SynthKorean path doesn't exist!")

        for file in os.listdir(dataset_path):
            if file.endswith(".jpg"):
                img_name, _ = os.path.splitext(file)
                img_path = os.path.join(dataset_path, file)

                self.img_paths.append(img_path)
                self.gt_paths.append(os.path.join(dataset_path, img_name + ".txt"))


    def __len__(self):
        return len(self.img_paths)

    def get_imagename(self, index):
        return self.img_paths[index]

    def load_image_gt_and_confidencemask(self, index):
        img_path = os.path.join(self.dataset_path, self.img_paths[index])
        gt_path = os.path.join(self.dataset_path, self.gt_paths[index])

        image = self.load_image(img_path)
        image_height, image_width = image.shape[0], image.shape[1]

        word_bboxes, are_valid_words, words = self.load_gt(gt_path, image_height, image_width)
        word_bboxes = np.float32(word_bboxes)

        confidence_mask = np.ones((image.shape[0], image.shape[1]), np.float32)
        character_bboxes = []
        new_words = []
        confidences = []
        if len(word_bboxes) > 0:
            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    print("there are empty words!!")
                    cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (0))

            for i in range(len(word_bboxes)):
                if words[i] == '###' or len(words[i].strip()) == 0:
                    continue

                pursedo_bboxes, bbox_region_scores, confidence = self.inference_pursedo_bboxes(self.net, image,
                                                                                               word_bboxes[i],
                                                                                               words[i])

                confidences.append(confidence)
                cv2.fillPoly(confidence_mask, [np.int32(word_bboxes[i])], (confidence))
                new_words.append(words[i])
                character_bboxes.append(pursedo_bboxes)

        return image, character_bboxes, new_words, confidence_mask, confidences, img_path

    def load_gt(self, gt_path, image_height, image_width):
        '''loads gt label

                :return

                word_bboxes[num_words x 4 x 2] - a list of character bounding box, each character bounding box
                represented as a 2D-ndarray[4 x 2]. num_chars is the number of characters in the word, so it is
                going to dependent on the lenght of each word. Starts at left-lower corner and goes counter-clock wise
                    Type: list(ndarray(shape=(4, 2)))

                text_tags[num_words] - boolean list indicating whether the word is valid or is "###" (DO NOT CARE)

                words[num_words] - a list of words within a single image
                    Type: list(str())
        '''
        text_tags = []
        word_bboxes = []
        words = []

        norm = math.sqrt(image_height * image_height + image_width * image_width)

        with open(gt_path, 'r', encoding='utf-8',) as f:
            for line in f.readlines():
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = line.replace('\ufeff', '')
                line = line.strip()
                splits = line.split(" ")

                cls, x, y, w, h, angle = list(map(float, splits[:6]))

                if angle < -50:
                    print("Min angle")
                    angle = 0

                rect = ((x * image_width, y * image_height), (w * norm, h * norm), angle * 180 / math.pi)
                pts = cv2.boxPoints(rect)

                word = ''
                delim = ''
                for t in splits[6:]:
                    word += delim + t
                    delim = ' '

                pts = pts.reshape(-1, 2)
                word_bboxes.append(pts)
                words.append(word.strip())

                if word == '*' or word.startswith('###'):  # or (w < h):
                    text_tags.append(True)
                else:
                    text_tags.append(False)

            return word_bboxes, text_tags, words

        """
        load_annotation()

        words = []

        with open(gt_path) as gt_file:
            lines = gt_file.readlines()

        image_diagonal = np.hypot([image_width], [image_height])[0]

        for line in lines:

            # remove trailing new line char
            if line.endswith("\n"):
                line = line[:len(line)-1]

            content_list = line.split(" ")
            assert len(content_list) == 7, "corrupt GT file: " + gt_path

            # 6 items each line mean: Lang,centerX,centerY,width,height,angle,transcripion
            # width and height are divided by image diagonal.
            # angle is in radians centered in the center of the bounding box
            # source: https://github.com/MichalBusta/E2E-MLT/issues/46
            box_center_x = np.float32(line[1]) * image_width
            box_center_y = np.float32(line[2]) * image_height
            box_width = np.float32(line[3]) * image_diagonal
            box_height = np.float32(line[4]) * image_diagonal
            angle = np.float32(line[5])
            word = line[6]
        """

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

if __name__ == '__main__':
    import torch
    from model import craft

    net = craft.CRAFT(use_vgg16_pretrained=False, freeze=True)
    net = torch.nn.DataParallel(net, device_ids=[0])

    state_dict = torch.load('/DATA1/isaac/CRAFT-Reimplemetation/pretrain/official_pretrained/craft_mlt_25k.pth')
    net.load_state_dict(state_dict)

    net = net.cuda()

    dataloader = SynthKorean(net,'/DATA1/isaac/ocr_data/Korean', viz=True, data_corruption=False)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    total = 0
    for index, (opimage, region_scores, affinity_scores, confidence_mask, confidences_mean, unnormalized_images, img_paths) in enumerate(train_loader):
        total += 1