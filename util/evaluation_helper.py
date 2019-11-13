from collections import namedtuple
from eval import rrc_evaluation_funcs
import importlib
import zipfile
import os


def get_recall(gt_char_bboxes, my_char_bboxes):
    for

    intersection / ground_truth_area

def get_precision():
    intersection / my_area



def points_to_polygon(points):
    """
    Returns a Polygon object to use with the Polygon2 class
    from a list of

    :param points: a list of 8 points (which is in [x1,y1,x2,y2,x3,y3,x4,y4]) that represent a quadrilateral or
    4 points (which is in [x1, y1, x3, y3]) that represent a rectangle
    :type points: list 8 or 4 ints
    :return: Polygon object
    """

    resBoxes = np.empty([1, 8], dtype='int32')

    if len(points) == 4:
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[0])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[2])
        resBoxes[0, 6] = int(points[3])
        resBoxes[0, 3] = int(points[2])
        resBoxes[0, 7] = int(points[1])

    if len(points) == 8:
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])

    else:
        raise ValueError("number of points should either be 4 (for a rectangle)"\
                         "or 8 (for x1,y1,x2,y2,x3,y3,x4,y4)")

    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)

def get_intersection(gt_poly, my_poly):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()

def get_union(gt_poly, my_poly):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)

