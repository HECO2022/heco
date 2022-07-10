from models.facealignment.FastFaceAlignemt import generateFastFan
import cv2
import dlib

import argparse
import time

from models.facedetect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from models.facedetect.vision.ssd.mb_tiny_fd import create_mb_tiny_fd_predictor

parser = argparse.ArgumentParser(
    description='detect_video')

parser.add_argument('--model_path', default="XXXXXXX", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--image_path', default="XXXXXX", type=str,
                    help='video dir')
parser.add_argument('--save_path', default="XXXXXX", type=str,
                    help='ouput video dir')
parser.add_argument('--test_device', default=0, type=int,
                    help='cuda:1')
args = parser.parse_args()





if __name__ == '__main__':


    landmarksDetector = generateFastFan(modelPath=args.model_path, deviceID=args.test_device)

    img = cv2.imread(args.image_path)

    #face detector
    candidate_size = 2000
    threshold = 0.99
    label_path = "XXXXXXX"
    class_names = [name.strip() for name in open(label_path).readlines()]
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=0)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=args.test_device)
    model_path = "XXXXXXXX"
    net.load(model_path)

    boxes, labels, probs = predictor.predict(img, candidate_size / 2, threshold)
    # print('boxes.shape:{},labels.shape:{},probs.shape:{}'.format(boxes.shape,labels.shape,probs.shape))
    print('boxes:',boxes)
    print('labels:',labels)
    print('probs:',probs)
    faces = []
    for box in boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        faces.append([box[0]-0.1*w, box[1]+0.05*w, box[2]+0.1*h, box[3]+0.1*h])
        cv2.rectangle(img, (box[0]-0.1*w, box[1]+0.1*w), (box[2]+0.1*h, box[3]+0.1*h), (0, 255, 0), 4)

    #detect landmarks
    landmarks = landmarksDetector.getLandmarksFromFrame(img,faces)
    #print('landmarks:',landmarks.shape)
    print(len(landmarks))
    for facel in landmarks:
        for point in facel:
            cv2.circle(img, (int(point[0]), int(point[1])), radius=1, color=(0, 0, 255), thickness=4)
    cv2.imwrite(args.save_path, img)




