"""
    Name: Ross Monaghan
    File: demo.py
    Description: File containing demonstration of gaze estimation functionality
    Date: 15/05/21

    ** THE FOLLOWING CODE CONTAINS SECTIONS FROM THE MSI-NET GITHUB REPOSITORY **
    ** URL: https://github.com/alexanderkroner/saliency **

    @InProceedings{Ruiz_2018_CVPR_Workshops,
    author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
    title = {Fine-Grained Head Pose Estimation Without Keypoints},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2018}
    }

"""

import cv2
import tensorflow as tf
import numpy as np
import os

import data
import utils
import hopenet

IMAGE_LIST = ["image1.jpg", "image2.jpg", "image3.jpg"]
INPUT_SIZE = 64
BIN_NUM = 66

RESULTS_PATH = "results/"
HPE_MODEL_PATH = "models/biwi_model_pretrained.h5"
SALIENCY_MODEL_PATH = "models/model_osie_cpu.pb"

face_cascade = cv2.CascadeClassifier('./haar_features/haarcascade_frontalface_default.xml')


# Method to predict a persons gaze direction
def predict_pose(img_file, frame):
    face_rects = face_cascade.detectMultiScale(image=frame, scaleFactor=1.2, minNeighbors=5)  # detecting faces

    if len(face_rects) > 0:
        cropped_face, loc = utils.crop_face(frame, face_rects, INPUT_SIZE)
        cropped_face = np.expand_dims(cropped_face, 0)

        pred_cont_yaw, pred_cont_pitch = net.test(cropped_face)

        masked_img, dir_len, head_ctr, dir_pt = utils.create_mask(frame, pitch=pred_cont_pitch, yaw=pred_cont_yaw,
                                                                  ptx=loc[0], pty=loc[1], size=100000,
                                                                  theta=20)
        cv2.imwrite("results/hpe/" + img_file, masked_img)

        return dir_len, head_ctr, dir_pt


# Method to estimate salient areas in an image
def estimate_saliency():
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

    iterator = data.get_dataset_iterator("results/hpe/")

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
            filename += ".jpg"

            try:
                with open("results/saliency/" + filename, "wb") as file:
                    file.write(output_file)
                print("Done!")
            except:
                print("Failed to write file.")


net = hopenet.HopeNet(num_bins=BIN_NUM, model_path=HPE_MODEL_PATH, new=False)
frames = []
dirs = []
heads = []

for image in IMAGE_LIST:
    frame = cv2.imread("images/" + image)
    frames.append(frame)

    dir_len, head_ctr, dir_pt = predict_pose(image, frame)
    dirs.append(dir_len)
    heads.append(head_ctr)

tf.compat.v1.disable_eager_execution()
estimate_saliency()

for i, image in enumerate(IMAGE_LIST):
    sal_img = cv2.imread("results/saliency/" + image)
    most_salient = utils.most_salient_area(dirs[i], heads[i], sal_img)

    final_img = cv2.addWeighted(most_salient, 0.6, frames[i], 0.5, 0.0)
    cv2.imwrite("results/predictions/" + image, final_img)