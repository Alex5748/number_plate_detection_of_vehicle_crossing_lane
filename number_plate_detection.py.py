import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from yolov5_tflite_image_inference import detect_image
from database_run import db

min_width_rect =150 #300  #150
min_height_rect =100 #300   #100

# initialize algorithm
algo = cv2.createBackgroundSubtractorMOG2(
    detectShadows=False, varThreshold=50)# var threshold=50


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    canny = cv2.Canny(blur, 5, 150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(76, 453), (72, 24), (631, 17), (631, 466)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return line_image


def start():
    BASE_PATH = os.getcwd()

    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        cropped_image = region_of_interest(frame)
        canny_image = canny(cropped_image)
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 400,
                                np.array([]), minLineLength=40, maxLineGap=50)
        line_image = display_lines(frame, lines)
        combo_image = cv2.addWeighted(frame, 1, line_image, 1, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(frame, (3, 3), 5)

        # apply on all frame
        image_sub = algo.apply(cropped_image)
        dilat = cv2.dilate(image_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        countour_shape, h = cv2.findContours(
            dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("combo_image", image_sub)
        allValues = []
        for (i, c) in enumerate(countour_shape):
            (x, y, w, h) = cv2.boundingRect(c)
            valid_vech = (w >= min_width_rect) and (h >= min_height_rect)
            if not valid_vech:
                continue

            rectangle_draw = cv2.rectangle(
                combo_image, (x, y), (x+h, y+h), (0, 255, 0), 2)

            #roi= (x, y, x+h, y+h)
            print("x+h", x+h)

            # rectangle box formed right side line
            line1 = cv2.line(combo_image, (x+h, y), (x+h, y+h), (255, 0, 0))
            x5 = 500
            y5 = 500
            # x2 = 360
            # y2 = 25
            x2 = 400
            y2 = 25
            # fixed lane on the road(coordinated manually given)
            draw_line1 = cv2.line(combo_image, (x5, y5),
                                  (x2, y2), (255, 0, 0,), 1)
            draw_line5=cv2.line(combo_image,(x5+20,y5),(x2+20,y2),(255,0,0),1)
            try:
                if(x+h > x5):
                    print("vehicle crossed lane")
                    vehicle_image = combo_image[y:y+h, x:x+h]
                    # cv2.imwrite('.jpg',vehicle_image)
                    # vehicle_image.save("lanecrossed.jpg")

                    # save path for cropped image
                    save_dir = BASE_PATH+'\Lane'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    print("Writing")
                    cv2.imshow("cropped_image", vehicle_image)
                    cv2.imwrite(os.path.join(
                        save_dir, 'cropped_image.jpg'), vehicle_image)
                    try:
                        print("Testing")
                        detect_image(weights=BASE_PATH+"/models/custom_plate.tflite", labels=BASE_PATH+"/labels/plate.txt", conf_thres=0.25, iou_thres=0.45,
                                     image_url=BASE_PATH+"/Lane/cropped_image.jpg", img_size=640)
                        print("exiting plate")

                        value = detect_image(weights=BASE_PATH+"/models/character.tflite", labels=BASE_PATH+"/labels/number.txt", conf_thres=0.25, iou_thres=0.45,
                                             image_url=BASE_PATH+"/output/cropped/1.jpg", img_size=640)
                        print("value", value)
                        if value is not None:
                            allValues.append(value)

                        # print("value", value)
                    except:
                        print("error")
                else:
                    print("lane is not crossed")
            except:
                break
        print("allValues", allValues)
        
        unique_list = list((allValues))
        print("uniques_list", unique_list)
        for i in unique_list:
            db(i)
        cv2.imshow("result", combo_image)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start()
