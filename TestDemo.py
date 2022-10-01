import cv2
from FaceAlignment.Models import ShapeRegressor
from FaceAlignment.Utils import BoundingBox

def main() -> None:
    test_images, test_bounding_box = [], []
    test_img_num = 507
    initial_number = 20
    landmark_num = 29

    for i in range(test_img_num):
        img_name = ''
        image = cv2.read.imread(img_name)
        test_images.append(image)

    bb_test_file_path = ''

    with open(bb_test_file_path) as f:
        for line in f.readlines():
            start_x, start_y, width, height = [int(x) for x in line.strip().split()]

            test_bounding_box.append(BoundingBox(start_x, 
                                                 start_y, 
                                                 width, 
                                                 height, 
                                                 start_x + width / 2.,
                                                 start_y + height / 2.))

    model_path = ''
    regressor = ShapeRegressor.load(model_path)

    while True:
        index = int(input("Image index: "))

        current_shape = regressor.predict(test_images[index],
                                          test_bounding_box[index],
                                          initial_number)

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
            

