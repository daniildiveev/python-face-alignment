from typing import List

import numpy as np
import cv2

from FaceAlignment import BoundingBox

def project_shape(shape:np.ndarray, bounding_box:BoundingBox) -> np.ndarray:
    temp = np.zeros(shape.shape[0], 2)
    
    for j in range(shape.shape[0]):
        temp[j][0] = (shape[j][0] - bounding_box.centroid_x) / (bounding_box.width / 2.)
        temp[j][1] = (shape[j][1] - bounding_box.centroid_y) / (bounding_box.height / 2.)

    return temp

def reproject_shape(shape:np.ndarray, bounding_box:BoundingBox) -> np.ndarray:
    temp = np.zeros(shape.shape[0], 2)

    for j in range(shape.shape[0]):
        temp[j][0] = (shape[j][0] * bounding_box.width / 2. + bounding_box.centroid_x)
        temp[j][1] = (shape[j][1] * bounding_box.height / 2. + bounding_box.centroid_y)

    return temp

def get_mean_shape(shapes:List[np.ndarray], bounding_boxes:List[BoundingBox]) -> np.ndarray:
    result = np.zeros(shapes[0].shape[0], 2)

    for i in range(len(shapes)):
        result += project_shape(shapes[i], bounding_boxes[i])

    return 1. / len(shapes) * result

def similarity_transform(shape1:np.ndarray,
                         shape2:np.ndarray,
                         rotation:np.ndarray) -> None:

    center_x_1, center_y_1, center_x_2, center_y_2 = (0, 0, 0, 0)

    for i in range(shape1.shape[0]):
        center_x_1 += shape1[i][0]
        center_y_1 += shape1[i][1]
        center_x_2 += shape2[i][0]
        center_y_2 += shape2[i][1]

    center_x_1 /= shape1.shape[0]
    center_y_1 /= shape1.shape[0]
    center_x_2 /= shape2.shape[0]
    center_y_2 /= shape2.shape[0]

    temp1, temp2 = np.copy(shape1), np.copy(shape2)

    for i in range(shape1.shape[0]):
        temp1[i][0] -= center_x_1
        temp1[i][1] -= center_y_1
        temp2[i][0] -= center_x_2
        temp2[i][1] -= center_y_2
    
    covar1, m1 = cv2.calcCovarMatrix(temp1, cv2.cv.CV_COVAR_COLS)
    covar2, m2 = cv2.calcCovarMatrix(temp2, cv2.cv.CV_COVAR_COLS)

    s1 = np.linalg.norm(covar1)
    s2 = np.linalg.norm(covar2)

    scale = s1 / s2
    
    temp1 = 1. / s1 * temp1
    temp2 = 1. / s2 * temp2

    num, den = 0, 0

    for i in range(shape1.shape[0]):
        num += temp1[i][1] * temp2[i][0] - temp1[i][0] * temp2[i][1]
        den += temp1[i][0] * temp2[i][0] + temp1[i][1] * temp2[i][1]

    norm = np.sqrt(num ** 2 + den ** 2)
    sin_theta = num / norm
    cos_theta = den / norm

    rotation[0][0] = cos_theta
    rotation[0][1] = -sin_theta
    rotation[1][0] = sin_theta
    rotation[1][1] = cos_theta