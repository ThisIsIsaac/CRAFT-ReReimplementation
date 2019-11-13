from util.file_utils import *
from util.data_loader import craft_base_dataset, random_scale
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import logging
import math
import csv

class NumberPlate(craft_base_dataset):
    def __init__(self, img_dir, target_size=768, viz=False, debug=False):
        super(NumberPlate, self).__init__(target_size, viz, debug)

        assert os.path.exists(img_dir), "Given path doesn't exist"

        # read GT file
        csv_gt_exists = False
        for file in os.listdir(img_dir):
            if file.endswith("CSV") or file.endswith("csv"):
                if csv_gt_exists:
                    raise ValueError("There is more than 1 csv file.")
                csv_gt_exists = True
                self.gt_path = os.path.join(img_dir, file)

        if not csv_gt_exists:
            raise FileNotFoundError("No ground truth CSV file found")

        self.gt = self.read_gt(self.gt_path)


    def read_gt(self, csv_path):

        def parse_char_bboxes(all_ponits_x, all_points_y):



        if not os.path.exists(csv_path):
            raise FileNotFoundError("No such GT file as " + csv_path)

        gt = {}

        with open(csv_path, mode="r") as gt_file:
            csv_reader = csv.DictReader(gt_file)
            for row in csv_reader:
                img_name = row["filename"]
                num_char_bboxes = row["region_count"]
                char_bboxes = row[]



    def read_img(self):
