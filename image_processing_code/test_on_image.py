import cv2
import dlib
import tensorflow as tf
import numpy as np
import os

import data
import utils
from imutils import face_utils
import hopenet

IMAGE_FILE = "images/frame_00139_rgb.png"
INPUT_SIZE = 64
BIN_NUM = 66

RESULTS_PATH = "results/"
FACE_LANDMARK_PATH = "haar_features/shape_predictor_68_face_landmarks.dat"
HPE_MODEL_PATH = "models/biwi_model_pretrained.h5"
SALIENCY_MODEL_PATH = "models/model_osie_cpu.pb"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FACE_LANDMARK_PATH)

net = hopenet.HopeNet(num_bins=BIN_NUM, model_path=HPE_MODEL_PATH, new=False)

frame = cv2.imread(IMAGE_FILE)

def predict_pose(frame):

    face_rects = detector(frame)

    if len(face_rects) > 0:
        shape = predictor(frame, face_rects[0])
        shape = face_utils.shape_to_np(shape)

        cropped_face = utils.crop_face_loosely(shape, frame, INPUT_SIZE)
        cropped_face = np.expand_dims(cropped_face, 0)

        pred_cont_yaw, pred_cont_pitch = net.test(cropped_face)

        masked_img = utils.create_mask(frame, pitch=pred_cont_pitch, yaw=pred_cont_yaw, ptx=shape[30][0], pty=shape[30][1], size=1000, theta=15)
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite("results/hpe/hpe_img.jpg", masked_img)


def test_model():
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.

    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    iterator = data.get_dataset_iterator("results/hpe/hpe_img.jpg")

    next_element, init_op = iterator

    input_images, original_shape, file_path = next_element

    graph_def = tf.compat.v1.GraphDef()

    with tf.compat.v1.gfile.Open(SALIENCY_MODEL_PATH, "rb") as file:
        graph_def.ParseFromString(file.read())

    [predicted_maps] = tf.compat.v1.import_graph_def(graph_def,
                                           input_map={"input": input_images},
                                           return_elements=["output:0"])

    jpeg = data.postprocess_saliency_map(predicted_maps[0],
                                         original_shape[0])

    print(">> Estimating Saliency...")

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        while True:
            try:
                output_file, path = sess.run([jpeg, file_path])
            except tf.compat.v1.errors.OutOfRangeError:
                break

            path = path[0][0].decode("utf-8")

            filename = os.path.basename(path)
            filename = os.path.splitext(filename)[0]
            filename += ".jpeg"

            # os.makedirs(paths["images"], exist_ok=True)

            try:
                with open("results/saliency/" + filename, "wb") as file:
                    file.write(output_file)
                print("Done!")
            except:
                print("Failed to write file.")


predict_pose(frame)

tf.compat.v1.disable_eager_execution()

test_model()