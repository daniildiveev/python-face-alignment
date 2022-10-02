from typing import List

import numpy as np
import cv2

class BoundingBox:
    def __init__(self,
                 start_x:int, 
                 start_y:int,
                 width:int,
                 height:int,
                 centroid_x:int,
                 centroid_y:int) -> None:
        self.start_x = start_x or 0
        self.start_y = start_y or 0
        self.width = width or 0
        self.height = height or 0
        self.centroid_x = centroid_x or 0
        self.centroid_y = centroid_y or 0

def project_shape(shape:np.ndarray, bounding_box:BoundingBox) -> np.ndarray:
    shape = np.array(shape)
    temp = np.zeros((shape.shape[0], 2))
    
    for j in range(shape.shape[0]):
        temp[j][0] = (shape[j][0] - bounding_box.centroid_x) / (bounding_box.width / 2.)
        temp[j][1] = (shape[j][1] - bounding_box.centroid_y) / (bounding_box.height / 2.)

    return temp

def reproject_shape(shape:np.ndarray, bounding_box:BoundingBox) -> np.ndarray:
    temp = np.zeros((shape.shape[0], 2))

    for j in range(shape.shape[0]):
        temp[j][0] = (shape[j][0] * bounding_box.width / 2. + bounding_box.centroid_x)
        temp[j][1] = (shape[j][1] * bounding_box.height / 2. + bounding_box.centroid_y)

    return temp

def get_mean_shape(shapes:List[np.ndarray], bounding_boxes:List[BoundingBox]) -> np.ndarray:
    result = np.zeros((shapes[0].shape[0], 2))

    for i in range(len(shapes)):
        result += project_shape(shapes[i], bounding_boxes[i])

    return 1. / len(shapes) * result

def similarity_transform(shape1:np.ndarray,
                         shape2:np.ndarray,) -> None:

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
    
    covar1, _ = cv2.calcCovarMatrix(temp1, np.zeros(len(temp1)).T, flags=cv2.COVAR_COLS)
    covar2, _ = cv2.calcCovarMatrix(temp2, np.zeros(len(temp1)).T, flags=cv2.COVAR_COLS)

    s1 = np.linalg.norm(covar1)
    s2 = np.linalg.norm(covar2)

    scale = s1 / s2
    
    temp1 = 1. / s1 * temp1
    temp2 = 1. / s2 * temp2

    num, den = 0, 0

    for i in range(shape1.shape[0]):
        num += temp1[i][1] * temp2[i][0] - temp1[i][0] * temp2[i][1]
        den += temp1[i][0] * temp2[i][0] + temp1[i][1] * temp2[i][1]

    norm = np.sqrt(num ** 2 + den ** 2 + 1e-10)
    sin_theta = num / norm
    cos_theta = den / norm

    rotation = np.zeros((2, 2))

    rotation[0][0] = cos_theta
    rotation[0][1] = -sin_theta
    rotation[1][0] = sin_theta
    rotation[1][1] = cos_theta

    return rotation, scale

def calculate_covarience(v1:np.ndarray,
                         v2:np.ndarray):

    v1 = v1 - np.mean(v1, dtype=np.float64)
    v2 = v2 - np.mean(v2, dtype=np.float64)

    return np.mean(np.dot(v1, v2))