import cv2
from FaceAlignment.Models import *
from FaceAlignment.Utils import BoundingBox

def main() -> None:
    img_num = 1345
    candidate_pixel_num = 400
    fern_pixel_num = 5
    first_level_num = 10
    second_level_num = 500
    landmark_num = 29
    initial_number = 20

    images, bounding_boxes, ground_truth_shapes = [], [], []

    img_dir = ''

    for i in range(img_num):
        img_name = ''
        image = cv2.imread(img_name)
        images.append(image)

    bb_file_path = ''
    
    with open(bb_file_path) as f:
        for line in f.readlines():
            start_x, start_y, width, height = [int(x) for x in line.strip().split()]

            bounding_boxes.append(BoundingBox(start_x, 
                                              start_y, 
                                              width, 
                                              height, 
                                              start_x + width / 2.,
                                              start_y + height / 2.))

    keypoints_file_path = ''

    with open(keypoints_file_path) as f:
        indexes_0 = f.readlines[:landmark_num]
        indexes_1 = f.readlines[landmark_num:]

        indexes_0 = [int(x) for x in indexes_0]
        indexes_1 = [int(x) for x in indexes_1]

        landmarks = np.array(zip(indexes_0, indexes_1))
        ground_truth_shapes.append(landmarks)

    regressor = ShapeRegressor(first_level_num, bounding_boxes)
    regressor.train(images, ground_truth_shapes, 
                    second_level_num, candidate_pixel_num, initial_number)

    regressor.save("./")

if __name__ == '__main__':
    main()