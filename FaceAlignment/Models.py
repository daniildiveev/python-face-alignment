from typing import List

import cv2
import numpy as np

class BoundingBox:
    def __init__(self):
        self.start_x = 0
        self.start_y = 0
        self.width = 0
        self.height = 0
        self.centroid_x = 0
        self.centroid_y = 0

class Fern:
    def __init__(self,
                 fern_pixel_num:int,) -> None:
        self.__fern_pixel_num = fern_pixel_num
        
    def train(self,
              candidate_pixel_intensity:List[np.ndarray],
              covariance:np.ndarray,
              candidate_pixel_locations:np.ndarray,
              nearest_landmark_index:np.ndarray, 
              regression_targets:List[np.ndarray]) -> List[np.ndarray]:
        self.__landmark_num = regression_targets[0].shape[0]
        self.__selected_pixel_index = np.zeros(self.__fern_pixel_num, 2)
        self.__selected_pixel_locations = np.zeros(self.__fern_pixel_num, 4)
        self.__selected_nearest_landmark_index = np.zeros(self.__fern_pixel_num, 2)
        
        self.__candidate_pixel_num = candidate_pixel_locations.shape[0]

        self.__threshold = np.zeros((self.__fern_pixel_num, 1))

        for i in range(self.__fern_pixel_num):
            random_direction = np.random.uniform(-1, 1, (self.__landmark_num, 2))
            random_direction = cv2.normalize(random_direction, random_direction)

            projection_result = np.zeros(len(regression_targets))

            for j in range(len(regression_targets)):
                projection_result[j] = np.sum(np.dot(regression_targets[j], random_direction))

            covariance_projection_density = np.zeros((self.__candidate_pixel_num, 1))

            for j in range(self.__candidate_pixel_num):
                covariance_projection_density[j] = np.cov(projection_result, candidate_pixel_intensity[j])

            max_correlation = -1
            max_pixel_index_1, max_pixel_index_2 = 0, 0

            for j in range(self.__candidate_pixel_num):
                for k in range(self.__candidate_pixel_num):
                    temp1 = covariance[j][j] + covariance[k][k] - 2 * covariance[j][k]

                    if abs(temp1) < 1e-10:
                        continue

                    flag = False

                    for p in range(i):
                        if ((j == self.__selected_pixel_index[p][0]) and (k == self.__selected_pixel_index[p][1])) or \
                            ((j == self.__selected_pixel_index[p][1]) and (k == self.__selected_pixel_index[p][0])):
                            flag = True
                            break
                    
                    if flag:
                        continue

                    temp = (covariance_projection_density[j] - covariance_projection_density[k]) / np.sqrt(temp1)

                    if abs(temp) > max_correlation:
                        max_correlation = temp
                        max_pixel_index_1 = j
                        max_pixel_index_2 = k

            self.__selected_pixel_index[i][0] = max_pixel_index_1
            self.__selected_pixel_index[i][1] = max_pixel_index_2
            self.__selected_pixel_locations[i][0] = candidate_pixel_locations[max_pixel_index_1][0]
            self.__selected_pixel_locations[i][1] = candidate_pixel_locations[max_pixel_index_1][1]
            self.__selected_pixel_locations[i][2] = candidate_pixel_locations[max_pixel_index_2][0]
            self.__selected_pixel_locations[i][4] = candidate_pixel_locations[max_pixel_index_2][1]
            self.__selected_nearest_landmark_index[i][0] = nearest_landmark_index[max_pixel_index_1]
            self.__selected_nearest_landmark_index[i][1] = nearest_landmark_index[max_pixel_index_2]

            max_diff = -1

            for j in range(len(candidate_pixel_intensity[max_pixel_index_1])):
                temp = candidate_pixel_intensity[max_pixel_index_2][j] - candidate_pixel_intensity[max_pixel_index_2][j]
                max_diff = max(temp, max_diff)

            self.__threshold[i] = np.random.uniform(-.2 * max_diff, .2*max_diff)
        
        bin_num = 2 ** self.__fern_pixel_num
        shapes_in_bin = [[] for _ in range(bin_num)]

        for i in range(len(regression_targets)):
            index = 0

            for j in range(self.__fern_pixel_num):
                density_1 = candidate_pixel_intensity[self.__selected_pixel_index[j][0]][i]
                density_2 = candidate_pixel_intensity[self.__selected_pixel_index[j][1]][i]

                if density_1 - density_2 >= self.__threshold[j]:
                    index += 2 ** j
                
            shapes_in_bin[index].append(i)

        shapes_in_bin = np.array(shapes_in_bin)
        prediction = [[] for _ in range(len(regression_targets))]
        self.__bin_outuput = [[] for _ in range(bin_num)]

        for i in range(bin_num):
            temp = np.zeros(self.__landmark_num, 2)
            bin_size = len(shapes_in_bin[i])

            for j in range(bin_size):
                index = shapes_in_bin[i][j]
                temp += regression_targets[index]

            if bin_size == 0:
                self.__bin_outuput[i] = temp
                continue

            temp = (1. / ((1. + 1000. / bin_size) * bin_size)) * temp
            self.__bin_outuput[i] = temp

            for j in range(bin_size):
                index = shapes_in_bin[i][j]
                prediction[index] = temp

        return prediction

    def predict(self, 
                image:np.ndarray,
                shape:np.ndarray,
                rotation:np.ndarray,
                bounding_box: BoundingBox,
                scale:float) -> np.ndarray:
        index = 0 

        for i in range(self.__fern_pixel_num):
            nearest_landmark_index_1 = self.__selected_nearest_landmark_index[i][0]
            nearest_landmark_index_2 = self.__selected_nearest_landmark_index[i][1]

            x = self.__selected_pixel_locations[i][0]
            y = self.__selected_pixel_locations[i][1]
            
            project_x = scale * (rotation[0][0] * x + rotation[0][1] * y) * bounding_box.width / 2. + shape[nearest_landmark_index_1][0]
            project_y = scale * (rotation[1][0] * x + rotation[1][1] * y) * bounding_box.height / 2. + shape[nearest_landmark_index_1][1]

            project_x = max(0., min(project_x, image.shape[1] - 1.))
            project_y = max(0., min(project_y, image.shape[0] - 1.))

            intensity_1 = int(image[int(project_y)][int(project_x)])

            x = self.__selected_pixel_locations[i][2]
            y = self.__selected_pixel_locations[i][3]

            project_x = scale * (rotation[0][0] * x + rotation[0][1]) * bounding_box.width / 2. + shape[nearest_landmark_index_2][0]
            project_y = scale * (rotation[1][0] * x + rotation[1][1]) * bounding_box.width / 2. + shape[nearest_landmark_index_2][1]

            intensity_2 = int(image[int(project_y)][int(project_y)])

            if intensity_1 - intensity_2 >= self.__threshold[i]:
                index += 2 ** i

        return self.__bin_outuput[index]
                           
    def write(self) -> None:
        raise NotImplementedError
    

class FernCascade:
    def __init__(self,
                 ferns_:List[Fern],
                 second_level_num_:int,) -> None:
        raise NotImplementedError

    def train(self, 
              images:List[np.ndarray],
              current_shapes:List[np.ndarray],
              ground_truth_shapes:List[np.ndarray],
              bounding_box:List[BoundingBox],
              mean_shape:np.ndarray, 
              second_level_num:int,
              candidate_level_pixel_num:int,
              fern_pixel_num:int, 
              curr_level_num:int,
              first_level_num:int,) -> List[np.ndarray]:
        raise NotImplementedError

    def predict(self,
                image:np.ndarray,
                bounding_box:BoundingBox,
                mean_shape:np.ndarray,
                shape:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def read(self) -> None:
        raise NotImplementedError

    def write(self) -> None:
        raise NotImplementedError


class ShapeRegressor:
    def __init__(self, 
                 first_level_num_:int,
                 landmark_num_:int,
                 fern_cascades_: List[FernCascade],
                 mean_shape_:np.ndarray, 
                 training_shapes_:List[np.ndarray], 
                 bounding_box_:List[BoundingBox],) -> None:
        raise NotImplementedError

    def train(self,
              images:List[np.ndarray],
              ground_truth_shapes:List[np.ndarray],
              bounding_box:List[BoundingBox],
              first_level_num:int,
              second_level_num:int,
              candidate_pixel_num:int, 
              fern_pixel_num:int,
              initial_num:int) -> None:
        raise NotImplementedError

    def predict(self,
                image:np.ndarray, 
                bounding_box:BoundingBox,
                initial_num:int,) -> np.ndarray:
        raise NotImplementedError

    def read(self) -> None:
        raise NotImplementedError

    def write(self) -> None:
        raise NotImplementedError

    def load(self, path:str) -> None:
        raise NotImplementedError

    def save(self, path:str) -> None:
        raise NotImplementedError