import cv2
import numpy as np
from FaceAlignment.Models import ShapeRegressor
from FaceAlignment.Utils import BoundingBox

def main() -> None:
    img_num = 1345
    candidate_pixel_num = 400
    fern_pixel_num = 5
    first_level_num = 10
    second_level_num = 10
    landmark_num = 29
    initial_number = 2

    images, bounding_boxes, ground_truth_shapes = [], [], []

    img_dir = '/content/data/trainingImages/'

    for i in range(1, img_num):
        img_name = img_dir + f"{i}.jpg"
        image = cv2.imread(img_name, 0)
        images.append(image)

    bb_file_path = '/content/data/boundingbox.txt'
    
    with open(bb_file_path) as f:
        for line in f.readlines():
            start_x, start_y, width, height = [int(x) for x in line.strip().split()]

            bounding_boxes.append(BoundingBox(start_x, 
                                              start_y, 
                                              width, 
                                              height, 
                                              start_x + width / 2.,
                                              start_y + height / 2.))

    keypoints_file_path = '/content/data/keypoints.txt'

    with open(keypoints_file_path) as f:
        for line in f.readlines():
            nums = [float(x) for x in line.split("\t")]
            x = nums[:landmark_num]
            y = nums[landmark_num:]

            landmarks = np.array(tuple(zip(x, y)))
            ground_truth_shapes.append(landmarks)

    regressor = ShapeRegressor(first_level_num, bounding_boxes)
    regressor.train(images, ground_truth_shapes, 
                    second_level_num, candidate_pixel_num, fern_pixel_num, initial_number)

    regressor.save("./model.pickle")

    test_images, test_bounding_box = [], []
    test_img_num = 507
    initial_number = 20
    landmark_num = 29

    img_dir = '/content/data/testImages/'

    for i in range(1, test_img_num):
        img_name = img_dir + f"{i}.jpg"
        image = cv2.imread(img_name, 0)
        test_images.append(image)

    bb_test_file_path = '/content/data/boundingbox_test.txt'

    with open(bb_test_file_path) as f:
        for line in f.readlines():
            start_x, start_y, width, height = [int(x) for x in line.strip().split()]

            test_bounding_box.append(BoundingBox(start_x, 
                                                 start_y, 
                                                 width, 
                                                 height, 
                                                 start_x + width / 2.,
                                                 start_y + height / 2.))

    while True:
        index = int(input("Image index: "))

        current_shape = regressor.predict(test_images[index],
                                          test_bounding_box[index],
                                          initial_number)

        print(current_shape)

        image_copy = test_images[index].copy()

        for i in range(landmark_num):
            image_copy = cv2.circle(image_copy,
                                    current_shape[i],
                                    3, (255, 0, 0),
                                    -1, 8, 0)

        cv2.imshow(image_copy)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()