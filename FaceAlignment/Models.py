from typing import List
import time

import cv2
import numpy as np
import pickle5 as pickle
from tqdm import trange

from FaceAlignment.Utils import *

class NotFittedError(Exception):
    pass

class Fern:
    def __init__(self,
                 fern_pixel_num:int,) -> None:
        self.__fern_pixel_num = fern_pixel_num
        self.__fitted = False
        
    def train(self,
              candidate_pixel_intensity:List[np.ndarray],
              covariance:np.ndarray,
              candidate_pixel_locations:np.ndarray,
              nearest_landmark_index:np.ndarray, 
              regression_targets:List[np.ndarray]) -> List[np.ndarray]:
        self.__landmark_num = regression_targets[0].shape[0]
        self.__selected_pixel_index = np.zeros((self.__fern_pixel_num, 2))
        self.__selected_pixel_locations = np.zeros((self.__fern_pixel_num, 4))
        self.__selected_nearest_landmark_index = np.zeros((self.__fern_pixel_num, 2))
        
        self.__candidate_pixel_num = candidate_pixel_locations.shape[0]

        self.__threshold = np.zeros((self.__fern_pixel_num, 1))

        for i in range(self.__fern_pixel_num):
            random_direction = np.random.uniform(-1, 1, (self.__landmark_num, 2))
            random_direction = cv2.normalize(random_direction, random_direction)
            projection_result = np.zeros(len(regression_targets))

            for j in range(len(regression_targets)):
                projection_result[j] = np.sum(regression_targets[j] * random_direction)

            covariance_projection_density = np.zeros((self.__candidate_pixel_num, 1))

            for j in range(self.__candidate_pixel_num):
                covariance_projection_density[j] = calculate_covarience(projection_result, candidate_pixel_intensity[j])

            max_correlation = -1
            max_pixel_index_1, max_pixel_index_2 = 0, 0

            for j in range(self.__candidate_pixel_num):
                for k in range(self.__candidate_pixel_num):
                    temp1 = covariance[j][j] + covariance[k][k] - 2 * covariance[j][k] 

                    if abs(float(temp1)) < 1e-10:
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
            self.__selected_pixel_locations[i][3] = candidate_pixel_locations[max_pixel_index_2][1]
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
                density_1 = candidate_pixel_intensity[int(self.__selected_pixel_index[j][0])][i]
                density_2 = candidate_pixel_intensity[int(self.__selected_pixel_index[j][1])][i]

                if density_1 - density_2 >= self.__threshold[j]:
                    index += 2 ** j
                
            shapes_in_bin[index].append(i)

        shapes_in_bin = np.array(shapes_in_bin, dtype=np.ndarray)
        prediction = [[] for _ in range(len(regression_targets))]
        self.__bin_outuput = [[] for _ in range(bin_num)]

        for i in range(bin_num):
            temp = np.zeros((self.__landmark_num, 2))
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
        
        self.__fitted = True

        return prediction

    def predict(self, 
                image:np.ndarray,
                shape:np.ndarray,
                rotation:np.ndarray,
                bounding_box: BoundingBox,
                scale:float) -> np.ndarray:
        
        if not self.__fitted:
            raise NotFittedError("")

        index = 0 

        for i in range(self.__fern_pixel_num):
            nearest_landmark_index_1 = int(self.__selected_nearest_landmark_index[i][0])
            nearest_landmark_index_2 = int(self.__selected_nearest_landmark_index[i][1])

            x = self.__selected_pixel_locations[i][0]
            y = self.__selected_pixel_locations[i][1]
            
            project_x = scale * (rotation[0][0] * x + rotation[0][1] * y) * bounding_box.width / 2. + shape[nearest_landmark_index_1][0]
            project_y = scale * (rotation[1][0] * x + rotation[1][1] * y) * bounding_box.height / 2. + shape[nearest_landmark_index_1][1]

            project_x = max(0., min(project_x, image.shape[1] - 1.))
            project_y = max(0., min(project_y, image.shape[0] - 1.))

            intensity_1 = int(image[int(project_y)][int(project_x)])

            x = self.__selected_pixel_locations[i][2]
            y = self.__selected_pixel_locations[i][3]

            project_x = scale * (rotation[0][0] * x + rotation[0][1] * y) * bounding_box.width / 2. + shape[nearest_landmark_index_2][0]
            project_y = scale * (rotation[1][0] * x + rotation[1][1] * y) * bounding_box.width / 2. + shape[nearest_landmark_index_2][1]

            print(int(project_x), int(project_y))
            intensity_2 = int(image[int(project_y)][int(project_y)])
        
            if intensity_1 - intensity_2 >= self.__threshold[i]:
                index += 2 ** i

        return self.__bin_outuput[index]
                

class FernCascade:
    def __init__(self,
                 fern_pixel_num:int,
                 second_level_num_:int,) -> None:
        self.__ferns = [Fern(fern_pixel_num) for _ in range(second_level_num_)]
        self.__second_level_num = second_level_num_
        self.__fitted = False

    def train(self, 
              images:List[np.ndarray],
              current_shapes:List[np.ndarray],
              ground_truth_shapes:List[np.ndarray],
              bounding_box:List[BoundingBox],
              mean_shape:np.ndarray, 
              candidate_pixel_num:int, 
              curr_level_num:int,
              first_level_num:int,) -> List[np.ndarray]:
        self.__candidate_pixel_locations = np.zeros((candidate_pixel_num, 2))
        self.__nearest_landmark_index = np.zeros((candidate_pixel_num, 1))
        self.regression_targets = [[] for _ in range(len(current_shapes))]

        for i in range(len(current_shapes)):
            self.regression_targets[i] = project_shape(ground_truth_shapes[i], bounding_box[i]) - \
                                         project_shape(current_shapes[i], bounding_box[i])

            rotation, scale = similarity_transform(mean_shape, 
                                                   project_shape(current_shapes[i], bounding_box[i]))

            rotation = rotation.T
            self.regression_targets[i] = scale * np.dot(self.regression_targets[i], rotation) 

        i = 0

        while i < candidate_pixel_num:
            x, y = np.random.uniform(-1., 1., size=(2,))

            if x ** 2 + y ** 2 < 1.:
                continue

            min_dist = 1e10
            min_index = 0

            for j in range(mean_shape.shape[0]):
                temp = (mean_shape[j][0] - x) ** 2 - (mean_shape[j][1] - y) ** 2

                if temp < min_dist:
                    min_dist = temp
                    min_index = j

            self.__candidate_pixel_locations[i][0] = x - mean_shape[min_index][0]
            self.__candidate_pixel_locations[i][1] = y - mean_shape[min_index][1]
            self.__nearest_landmark_index[i] = min_index

            i += 1

        denseties = [[] for _ in range(candidate_pixel_num)]

        for i in trange(len(images)):
            temp = project_shape(current_shapes[i], bounding_box[i])
            rotation, scale = similarity_transform(temp, mean_shape)

            for j in range(candidate_pixel_num):
                project_x = rotation[0][0] * self.__candidate_pixel_locations[j][0] + \
                    rotation[0][1] * self.__candidate_pixel_locations[j][1]
                project_y = rotation[1][0] * self.__candidate_pixel_locations[j][0] + \
                    rotation[1][1] * self.__candidate_pixel_locations[j][1]

                index = int(self.__nearest_landmark_index[j][0])

                real_x = int(project_x + current_shapes[i][index][0])
                real_y = int(project_y + current_shapes[i][index][1])
                real_x = int(max(0., min(real_x, images[i].shape[1] - 1.)))
                real_y = int(max(0., min(real_y, images[i].shape[0] - 1.)))

                denseties[j].append(images[i][real_y][real_x])

        denseties = np.array(denseties)
        covarience = np.cov(denseties)

        prediction = [np.zeros((mean_shape.shape[0], 2)) for _ in range(len(self.regression_targets))]

        if len(self.__ferns) != self.__second_level_num:
            raise ValueError("num ferns must be %s, got %s", (self.__second_level_num, len(self.__ferns)))

        t = time.time()

        for i in range(self.__second_level_num):
            temp = self.__ferns[i].train(denseties, 
                                         covarience,
                                         self.__candidate_pixel_locations, 
                                         self.__nearest_landmark_index, 
                                         self.regression_targets)
            
            for j in range(len(temp)):
                prediction[j] += temp[j]
                self.regression_targets -= temp[j]

            if (i + 1) % 5 == 0:
                print(f"Fern cascades: {curr_level_num} out of {first_level_num};") 
                print(f"Ferns: {i+1} out of {self.__second_level_num}")

                remaining_level_num = (first_level_num - curr_level_num) * 500 + self.__second_level_num - i
                time_remaining = 0.2 * (time.time() - t) * remaining_level_num

                print(f"Expected remaining time: {time_remaining // 60} min {int(time_remaining) % 60} sec")

        for i in range(len(prediction)):
            rotation, scale = similarity_transform(project_shape(current_shapes[i], bounding_box[i]), mean_shape)
            rotation = rotation.T
            prediction[i] = scale * np.dot(prediction[i], rotation)

        self.__fitted = True

        return prediction

    def predict(self,
                image:np.ndarray,
                bounding_box:BoundingBox,
                mean_shape:np.ndarray,
                shape:np.ndarray) -> np.ndarray:
        if not self.__fitted:
            raise NotFittedError("")

        result = np.zeros((shape.shape[0], 2))
        rotation, scale = similarity_transform(project_shape(shape, bounding_box),
                                               mean_shape)
        
        for fern in self.__ferns:
            result += fern.predict(image, shape, rotation, bounding_box, scale)

        rotation, scale = similarity_transform(project_shape(shape, bounding_box),
                                               mean_shape)
        
        rotation = rotation.T
        result = scale * np.dot(result, rotation)

        return result


class ShapeRegressor:
    def __init__(self, 
                 first_level_num_:int,
                 bounding_box_:List[BoundingBox],
                 landmark_num:int=None,
                 training_shapes:List[np.ndarray]=None,
                 mean_shape=None,
                 ferncascade_params:dict=None) -> None:
        self.__first_level_num = first_level_num_
        self.__bounding_box = bounding_box_
        self.landmark_num = landmark_num
        self.training_shapes = training_shapes
        self.mean_shape = mean_shape
        self.ferncascade_params = ferncascade_params

        self.__fitted = False

    def train(self,
              images:List[np.ndarray],
              ground_truth_shapes:List[np.ndarray],
              second_level_num:int,
              candidate_pixel_num:int, 
              fern_pixel_num:int,
              initial_num:int) -> None:
        print('Start training ...')

        self.__landmark_num = self.landmark_num or ground_truth_shapes[0].shape[0]
        self.__training_shapes = self.training_shapes or ground_truth_shapes.copy()
        images = np.array(images,dtype=np.ndarray)
        augmented_images = []
        augmented_bounding_box = []
        augmented_ground_truth_shapes = []
        self.current_shapes = []

        for i in range(len(images)):
            augmentation_indexes = np.random.randint(0, len(images) - 1, size=(initial_num,))
            
            for ai in augmentation_indexes:
                augmented_images.append(images[i])
                augmented_ground_truth_shapes.append(ground_truth_shapes[i])
                augmented_bounding_box.append(self.__bounding_box[i])

                temp = ground_truth_shapes[ai]
                temp = project_shape(temp, self.__bounding_box[ai])
                temp = reproject_shape(temp, self.__bounding_box[i])

                self.current_shapes.append(temp)

        self.__mean_shape = self.mean_shape or get_mean_shape(ground_truth_shapes, self.__bounding_box)

        self.__ferncascade_params = self.ferncascade_params or {
            'fern_pixel_num' : fern_pixel_num,
            'second_level_num_' : second_level_num
        }


        self.__fern_cascades = [FernCascade(**self.__ferncascade_params) for _ in range(self.__first_level_num)]
        self.current_shapes = np.array(self.current_shapes, dtype=np.ndarray)
        
        for i in range(self.__first_level_num):
            print(f"Training fern cascades: {i+1} out of {self.__first_level_num}")
            prediction = self.__fern_cascades[i].train(augmented_images,
                                                       self.current_shapes,
                                                       augmented_ground_truth_shapes, 
                                                       augmented_bounding_box,
                                                       self.__mean_shape,
                                                       candidate_pixel_num, 
                                                       i + 1,
                                                       self.__first_level_num)
        
            for j in range(len(prediction)):
                self.current_shapes[j] = prediction[j] + project_shape(self.current_shapes[j], augmented_bounding_box[j])
                self.current_shapes[j] = reproject_shape(self.current_shapes[j], augmented_bounding_box[j])
        
        self.__fitted = True

    def predict(self,
                image:np.ndarray,
                bounding_box:BoundingBox,
                initial_num:int,) -> np.ndarray:
        result = np.zeros((self.__landmark_num, 2))

        for i in range(initial_num):
            index = int(np.random.uniform(0, len(self.__training_shapes)))
            current_shape = self.__training_shapes[index]
            current_bounding_box = self.__bounding_box[index]

            current_shape = project_shape(current_shape, current_bounding_box)
            current_shape = reproject_shape(current_shape, bounding_box)

            for fern_cascade in self.__fern_cascades:
                prediction = fern_cascade.predict(image, 
                                                  bounding_box, 
                                                  self.__mean_shape,
                                                  current_shape)

                current_shape = prediction + project_shape(current_shape, bounding_box)
                current_shape = reproject_shape(current_shape, bounding_box)

            result += current_shape

        return 1. / initial_num * result

    @classmethod
    def load(cls, path:str) -> 'ShapeRegressor':
        with open(path, 'rb') as f:
            config = pickle.load(f)

        return cls(
            config['first_level_num'],
            config['bounding_boxes'],
            config['landmark_num'],
            config['training_shapes'],
            config['mean_shape'],
            config['ferncascade_config']
        )

    def save(self, path:str) -> None:
        data = {
            'first_level_num' : self.__first_level_num,
            'landmark_num' : self.__mean_shape.shape[0],
            'mean_shape' : [],
            'training_shapes' : [],
            'bounding_boxes' : [],
            'ferncascade_config' : self.__ferncascade_params,
        }

        data['mean_shape'] = [tuple(self.__mean_shape[i]) for i in range(self.__mean_shape.shape[0])]
        data['bounding_boxes'] = [BoundingBox(bounding_box.start_x,
                                   bounding_box.start_y,
                                   bounding_box.width,
                                   bounding_box.height,
                                   bounding_box.centroid_x,
                                   bounding_box.centroid_y) for bounding_box in self.__bounding_box]
        data['training_shapes'] = [self.__training_shapes[i].tolist() for i in range(len(self.__training_shapes))]
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)