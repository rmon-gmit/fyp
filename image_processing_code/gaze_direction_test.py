from time import sleep
import cv2
import dlib
import gaze_direction
import utils
from imutils import face_utils
import datasets
import hopenet

PROJECT_DIR = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\"

BIWI_DATA_DIR = "C:\\Users\\rossm\\kinect_head_pose_db\\hpdb\\"
BIWI_TEST_SAVE_DIR = PROJECT_DIR + "data\\biwi_test\\"

# MODEL_PATH = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\model\\alexnet_biwi_10_epoch.h5"
MODEL_PATH = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\model\\vgg16_biwi_1_epoch.h5"

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16

face_landmark_path = PROJECT_DIR + 'haar_features\\shape_predictor_68_face_landmarks.dat'

dataset = datasets.Biwi(BIWI_DATA_DIR, '\\filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.95)

net = hopenet.HopeNet(dataset, INPUT_SIZE, BIN_NUM, BATCH_SIZE, MODEL_PATH, new=False)

cap = cv2.VideoCapture(0)  # capturing video, will be replaced by a function get_frames which will get frames from a camera
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

INPUT_SIZE = 64

frames = []

while True:
    ret, frame = cap.read(0)  # getting the frame from the video, this will be replaced by the return contents of get_frames
    faces = detector(frame)

    if ret:
        face_rects = detector(frame, 0)
        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)
            face_crop = utils.crop_face_loosely(shape, frame, INPUT_SIZE)

            frames.append(face_crop)
            if len(frames) == 1:
                print(shape[30])
                pred_cont_yaw, pred_cont_pitch, pred_cont_roll = net.test(frames)

                cv2_img = utils.draw_axis(frame, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, tdx=shape[30][0],
                                          tdy=shape[30][1], size=100)
                cv2.imshow("cv2_img", cv2_img)
                frames = []

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
