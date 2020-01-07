import torch
import torch.utils.data as data
import scipy.io as scio
from util.gaussian import GaussianTransformer
from util.watershed import watershed
import re
import itertools
from util.file_utils import *
from util.mep import mep
import random
from PIL import Image
import torchvision.transforms as transforms
from util import craft_utils
import Polygon as plg
import numpy as np
import cv2
import util.imgproc as imgproc


def ratio_area(h, w, box):
    area = h * w
    ratio = 0
    for i in range(len(box)):
        poly = plg.Polygon(box[i])
        box_area = poly.area()
        tem = box_area / area
        if tem > ratio:
            ratio = tem
    return ratio, area

def rescale_img(img, box, h, w):
    image = np.zeros((768,768,3),dtype = np.uint8)
    length = max(h, w)
    scale = 768 / length           ###768 is the train image size
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    image[:img.shape[0], :img.shape[1]] = img
    box *= scale
    return image

def random_scale(img, char_bboxes, min_size):
    """Scales the original image fit within max and min dimension while applying random scailing, one of
    0.5, 1.0, 1.5, 2.0.

    Random scailing is applied in order to increase scale invariance. It is discussed in CRAFT paper section 4.4.

    :param img:
    :param char_bboxes:
    :param min_size:
    :return:
    """
    h, w = img.shape[0:2]
    scale = 1.0

    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)


    scale_values = np.array([0.5, 1.0, 1.5, 2.0])
    scale1 = np.random.choice(scale_values)

    if min(h, w) * scale * scale1 <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    else:
        scale = scale * scale1

    if type(char_bboxes) == list:
        for bboxes in char_bboxes:
            bboxes *= scale
    else:
        char_bboxes *= scale

    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    return img

def padding_image(image,imgsize, char_bboxes=None):
    length = max(image.shape[0:2])
    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)

    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image

    if char_bboxes is not None:
        if type(char_bboxes) == list:
            for bboxes in char_bboxes:
                bboxes *= scale
        else:
            char_bboxes *= scale

    return img, char_bboxes

def random_crop(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
            word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    #### IC15 for 0.6, MLT for 0.35 #####
    #if random.random() > 0.6 and len(word_bboxes) > 0:
    # Todo: this branch has bug. it doesn't adjust the word_bboxes while adjusting the image.
    if False:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)

        #### train for MLT dataset ###
        i, j = 0, 0
        crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx], _ = padding_image(imgs[idx], tw)

    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


class craft_base_dataset(data.Dataset):
    """
    Methods to override:
    * get_imagenmae
    * load_image_gt_and_confidencemask
    """
    def __init__(self, target_size=768, viz=False, debug=False, perform_input_data_corruption=True, out_path="saved_inputs"):
        self.target_size = target_size
        self.viz = viz
        self.debug = debug
        self.gaussianTransformer = GaussianTransformer(imgSize=1024, region_threshold=0.35, affinity_threshold=0.15)
        self.perform_input_data_corruption=perform_input_data_corruption
        self.out_path = out_path

        self.text_threshold=0.7
        self.link_threshold=0.4
        self.low_text=0.45

    def __getitem__(self, index):
        return self.pull_item(index)

    def load_image_gt_and_confidencemask(self, index):
        '''
        method to override.

        :return
        image[height x width x channel] - image as an ndarray
            type: ndarray

        char_bboxes[num_words x num_chars x 4 x 2] - a list of character bounding box, each character bounding box
        represented as a 3D-ndarray[num_chars x 4 x 2]. num_chars is the number of characters in the word, so it is
        going to dependent on the lenght of each word.
            Type: list(ndarray(shape=(num_char, 4, 2)))

        words[num_words] - a list of words within a single image
            Type: list(str())

        confidence_mask[height x width] - a map that represents the confidence for each pixel of the image. Confidence
        is a float where 1.0 is completely sure. For supervised learning,
        manually append 1.0 for each pixel becasue the character bboxes are tagged. For the weakly supervised method
        call craft_base_dataset.inference_pursedo_bboxes() method to produce confidence scores
            Type: ndarray(shape=(height, width))

        confidences[num_words] - confidence -- as float -- for each word's character bboxes. For supervised learning,
        manually append 1.0 for each word becasue the character bboxes are tagged. For the weakly supervised method
        call craft_base_dataset.inference_pursedo_bboxes() method to produce confidence scores

        image_path - path to image for debugging purposes
        '''
        return [], [], [], [], [], []

    def crop_image_by_bbox(self, image, box):
        w = (int)(np.linalg.norm(box[0] - box[1]))
        h = (int)(np.linalg.norm(box[0] - box[3]))
        width = w
        height = h
        if h > w * 1.5:
            width = h
            height = w
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[width, 0], [width, height], [0, height], [0, 0]])))
        else:
            M = cv2.getPerspectiveTransform(np.float32(box),
                                            np.float32(np.array([[0, 0], [width, 0], [width, height], [0, height]])))

        warped = cv2.warpPerspective(image, M, (width, height))
        return warped, M

    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len

    def inference_pursedo_bboxes(self, net, image, word_bbox, word):
        net.eval()
        with torch.no_grad():
            word_image, MM = self.crop_image_by_bbox(image, word_bbox)

            real_word_without_space = word.replace('\s', '')
            real_char_nums = len(real_word_without_space)
            input = word_image.copy()
            scale = 64.0 / input.shape[0]
            input = cv2.resize(input, None, fx=scale, fy=scale)

            img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406),
                                                                       variance=(0.229, 0.224, 0.225)))
            img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
            img_torch = img_torch.type(torch.FloatTensor).cuda()
            scores, _ = net(img_torch)
            region_scores = scores[0, :, :, 0].cpu().data.numpy()
            region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255)
            bgr_region_scores = cv2.resize(region_scores, (input.shape[1], input.shape[0]))
            bgr_region_scores = cv2.cvtColor(bgr_region_scores, cv2.COLOR_GRAY2BGR)
            pursedo_bboxes = watershed(input, bgr_region_scores, False)

            _tmp = []
            for i in range(pursedo_bboxes.shape[0]):
                if np.mean(pursedo_bboxes[i].ravel()) > 2:
                    _tmp.append(pursedo_bboxes[i])
                else:
                    print("filter bboxes", pursedo_bboxes[i])
            pursedo_bboxes = np.array(_tmp, np.float32)
            if pursedo_bboxes.shape[0] > 1:
                index = np.argsort(pursedo_bboxes[:, 0, 0])
                pursedo_bboxes = pursedo_bboxes[index]

            confidence = self.get_confidence(real_char_nums, len(pursedo_bboxes))

            bboxes = []
            if confidence <= 0.5:
                width = input.shape[1]
                height = input.shape[0]

                width_per_char = width / len(word)
                for i, char in enumerate(word):
                    if char == ' ':
                        continue
                    left = i * width_per_char
                    right = (i + 1) * width_per_char
                    bbox = np.array([[left, 0], [right, 0], [right, height],
                                     [left, height]])
                    bboxes.append(bbox)

                bboxes = np.array(bboxes, np.float32)
                confidence = 0.5

            else:
                bboxes = pursedo_bboxes

            bboxes /= scale

            for j in range(len(bboxes)):
                ones = np.ones((4, 1))
                tmp = np.concatenate([bboxes[j], ones], axis=-1)
                I = np.matrix(MM).I
                ori = np.matmul(I, tmp.transpose(1, 0)).transpose(1, 0)
                bboxes[j] = ori[:, :2]

            bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., image.shape[0] - 1)
            bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., image.shape[1] - 1)

            return bboxes, region_scores, confidence

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def get_imagename(self, index):
        return None

    def saveInput(self, image_name, image, region_scores, affinity_scores, confidence_mask, out_dir):

        imagename, _ = os.path.splitext(os.path.basename(image_name))
        imagename = imagename + "_after_processing.jpg"
        outpath = os.path.join(out_dir, imagename)

        craft_utils.save_outputs(image, region_scores, affinity_scores, self.text_threshold,
                                 self.link_threshold, self.low_text, outpath)

    def saveImage(self, imagename, image, bboxes, affinity_bboxes, region_scores, affinity_scores, confidence_mask, out_dir):
        output_image = np.uint8(image.copy())
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        if len(bboxes) > 0:
            affinity_bboxes = np.int32(affinity_bboxes)
            for i in range(affinity_bboxes.shape[0]):
                cv2.polylines(output_image, [np.reshape(affinity_bboxes[i], (-1, 1, 2))], True, (255, 0, 0))
            for i in range(len(bboxes)):
                _bboxes = np.int32(bboxes[i])
                for j in range(_bboxes.shape[0]):
                    cv2.polylines(output_image, [np.reshape(_bboxes[j], (-1, 1, 2))], True, (0, 0, 255))

        target_gaussian_heatmap_color = imgproc.cvt2HeatmapImg(region_scores / 255)
        target_gaussian_affinity_heatmap_color = imgproc.cvt2HeatmapImg(affinity_scores / 255)
        heat_map = np.concatenate([target_gaussian_heatmap_color, target_gaussian_affinity_heatmap_color], axis=1)
        confidence_mask_gray = imgproc.cvt2HeatmapImg(confidence_mask)
        output = np.concatenate([output_image, heat_map, confidence_mask_gray], axis=1)

        imagename, _ = os.path.splitext(os.path.basename(imagename))
        imagename = imagename + "_before_processing.jpg"
        outpath = os.path.join(out_dir, imagename)

        if not os.path.exists(os.path.dirname(outpath)):
            os.mkdir(os.path.dirname(outpath))
        cv2.imwrite(outpath, output)

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask, confidences, image_path = self.load_image_gt_and_confidencemask(index)

        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []

        if len(character_bboxes) > 0:
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                          character_bboxes,
                                                                                          words)
        if self.viz:
            self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinity_scores, confidence_mask, self.out_path)

        # perform random corruption (random scailing, crop, horizontal flip, rotation) in order to increase robustness
        if self.perform_input_data_corruption:
            image = random_scale(image, character_bboxes, self.target_size)
            random_transforms = [image, region_scores, affinity_scores, confidence_mask]
            random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
            random_transforms = random_horizontal_flip(random_transforms)
            random_transforms = random_rotate(random_transforms)
            image, region_scores, affinity_scores, confidence_mask = random_transforms

        # resize and pad images
        else:
            image, character_bboxes = padding_image(image, self.target_size, character_bboxes)
            region_scores, _ = padding_image(region_scores, self.target_size)
            affinity_scores, _ = padding_image(affinity_scores, self.target_size)
            confidence_mask, _ = padding_image(confidence_mask, self.target_size)

        # resize GT labels to put them in the same dimension as the output CRAFT.
        # CRAFT's output is h/2, w/2 ()
        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
        confidence_mask = self.resizeGt(confidence_mask)

        # the network's output is between 0~1
        region_scores = region_scores / 255
        affinity_scores = affinity_scores / 255

        if self.viz:
            self.saveInput(self.get_imagename(index), image, region_scores, affinity_scores, confidence_mask, self.out_path)

        # another data corruption on brightness and saturation
        # however, we always perform this regardless of the "perform_input_data_corruption" flag since this does not
        # change the dimensions.
        image = Image.fromarray(image)
        image = image.convert('RGB') # convert to RGB (why is this necessary? Isn't it already RGB?)
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)
        np_array_image = np.array(image)

        # convert preprocessed image to tensor
        image_tensor = imgproc.normalizeMeanVariance(np_array_image, mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image_tensor = torch.from_numpy(image_tensor).float().permute(2, 0, 1)

        region_scores_torch = torch.from_numpy(region_scores).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        return image_tensor, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences, np_array_image, image_path

if __name__ == '__main__':
    from data_loader.Synth80k import Synth80k

    dataloader = Synth80k('/DATA1/isaac/ocr_data/SynthText', viz=True, perform_input_data_corruption=True)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    total = 0

    for index, (image_tensor, region_scores, affinity_scores, confidence_mask, confidences_mean, unnormalized_images, img_paths) in enumerate(train_loader):
        total += 1


