from util.file_utils import *
from util.data_loader import craft_base_dataset, random_scale
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import logging
import math

class KAIST(craft_base_dataset):
    """
    dataset formats:
    * all image files end with ".JPG"
    * for every image file, there is a ".xml" file and ".bmp" file. The names of the two files are identical
    to the image file they describe
    * ".xml" file contains ground-truth: "imageName", "resolution", "words", and "character"

    Directory structure:
    KAIST
    ├── English
    |   ├── Digital_Camera
    |   |   ├──(E.S)A-shadow
    |   |   |   |   ├──  image1.PNG
    |   |   |   |   ├──  image1.xml
    |   |   |   |   ├──  image1.bmp
    |   |   |   |   ├──  image2.PNG
    |   |   |   |   ├──  image2.xml
    |   |   |   |   ├──  image2.bmp
    |   |   |   |  ...
    |   |   ├──(E.S)B-light
    |   |  ...
    |   └── Mobile_Phone
    ├── Korean
    |   ├── Digital_Camera
    |   └── Mobile_Phone
    └── Mixed
        ├── Digital_Camera
        └── Mobile_Phone
    """
    def __init__(self, path, target_size=768, viz=False, debug=False, data_corruption=False):
        super(KAIST, self).__init__(target_size, viz, debug, data_corruption)
        self.img_paths = []
        self.xml_paths = []
        self.target_size = target_size
        self.debug = debug
        self.skipped_images = 0

        assert len(os.listdir(path)) == 3, "make sure the path to KAIST dataset is correct"
        for lang_dir in os.listdir(path):
            lang_path = os.path.join(path, lang_dir)
            assert os.path.isdir(lang_path)
            assert len(os.listdir(lang_path)) == 2, "make sure the path to KAIST dataset is correct"

            for cam_dir in os.listdir(lang_path):
                cam_path = os.path.join(lang_path, cam_dir)
                assert os.path.isdir(cam_path)

                for img_dir in os.listdir(cam_path):
                    if img_dir.endswith(".zip"):
                        continue

                    img_dir_path = os.path.join(cam_path, img_dir)

                    for file in os.listdir(img_dir_path):
                        if file.endswith(".JPG") or file.endswith(".jpg"):
                            img_file = os.path.join(img_dir_path, file)
                            xml_file = img_file[:len(img_file)-len("jpg")] + "xml"

                            is_valid = self.is_valid(img_file, xml_file)
                            if is_valid:
                                self.img_paths.append(img_file)
                            else:
                                self.skipped_images += 1

        for img_path in self.img_paths:
            img_file_format = "jpg"
            xml_file_format = "xml"
            xml_path = img_path[:len(img_path) - len(img_file_format)] + xml_file_format
            self.xml_paths.append(xml_path)

        logging.info("while initializing KAIST dataset, " + str(self.skipped_images)
                     + " images were skipped due to data corruption or missing xml file")

    def __len__(self):
        return len(self.img_paths)

    def is_valid(self, img_file, xml_path):
        if not (os.path.exists(xml_path) and os.path.exists(img_file)):
            return False

        try:
            _ = cv2.imread(img_file, cv2.IMREAD_COLOR)
            _,_,_ = self.read_xml(xml_path)
            return True

        except:
            return False

    def read_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        resolution = root[0][1].attrib
        img_width = resolution["x"]
        img_height = resolution["y"]

        words = root[0][2]
        words_text = []
        char_bboxes = []
        for word in words:
            word_text = ""
            char_bbox = np.ndarray((len(word), 4, 2), np.float32)
            for char_idx, char in enumerate(word):
                char = char.attrib

                # upper-left corner, height, and width (from the corner)
                char_x = int(char["x"])
                char_y = int(char["y"])
                char_width = int(char["width"])
                char_height = int(char["height"])

                if math.isnan(char_x) or math.isnan(char_y) or math.isnan(char_width) or math.isnan(char_height):
                    raise ValueError("Corrupt data in file: " + xml_path +"\n Nan in one of the coordinates")

                # upper-left corner
                char_bbox[char_idx][0][0] = char_x
                char_bbox[char_idx][0][1] = char_y

                # upper-right corner
                char_bbox[char_idx][1][0] = char_x + char_width
                char_bbox[char_idx][1][1] = char_y

                # lower-right corner
                char_bbox[char_idx][2][0] = char_x + char_width
                char_bbox[char_idx][2][1] = char_y + char_height

                # lower-left corner
                char_bbox[char_idx][3][0] = char_x
                char_bbox[char_idx][3][1] = char_y + char_height

                word_text += char["char"]

            if len(word_text) != 0:
                words_text.append(word_text)
            if len(char_bbox) != 0:
                char_bboxes.append(char_bbox)

        return char_bboxes, words_text, (img_width, img_height)

    def read_img(self, img_path, char_bboxes):
        """Read image from disk. When read, by default, the bytes are arranged in BGR, not RGB.
        So we re-order the colors to RGB with cv2.cvtColor.

        source:https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor

        :param img_path:
        :param char_bboxes:
        :return:
        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_imagename(self, index):
        return self.img_paths[index].split("/")[-1]

    def load_image_gt_and_confidencemask(self, index):
        if index < 0 or index >= len(self):
            raise ValueError("index out of range")

        img_path = self.img_paths[index]
        xml_path = self.xml_paths[index]
        assert img_path[:len(img_path)-3] == xml_path[:len(xml_path)-3], \
            "Bug: something went wrong in KAIST __init__ when reading image and xml files."

        char_bboxes, words , _= self.read_xml(xml_path)
        img = self.read_img(img_path, char_bboxes)

        # since KAIST has character-level bboxes, confidence is 100 for all char bboxes.
        confidence = np.ones((len(words)), np.float32)
        confidence_mask = np.ones((img.shape[0], img.shape[1]), np.float32)

        return img, char_bboxes, words, confidence_mask, confidence, img_path

if __name__ == '__main__':
    import torch

    dataloader = KAIST('/DATA1/isaac/ocr_data/KAIST', viz=True, data_corruption=False)
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
