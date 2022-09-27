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
    
    
    