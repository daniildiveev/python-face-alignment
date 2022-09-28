from typing import List

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
                 fern_pixel_num:int, 
                 selected_nearest_landmark_index:np.ndarray,
                 threshold_:np.ndarray,
                 selected_pixel_index:np.ndarray,
                 selected_pixel_locations_:np.ndarray,
                 bin_output_:List[np.ndarray]) -> None:
        raise NotImplementedError

    def train(self,
              candidate_pixel_intensity:List[np.ndarray],
              covariance:np.ndarray,
              candidate_pixel_locations:np.ndarray,
              nearest_landmark_index:np.ndarray, 
              regression_targets:List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError

    def predict(self, 
                image:np.ndarray,
                shape:np.ndarray,
                rotation:np.ndarray,
                bounding_box: BoundingBox,
                scale:float) -> np.ndarray:
        raise NotImplementedError
    
    def read(self) -> None:
        raise NotImplementedError
        
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