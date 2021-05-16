"""
    Name: Ross Monaghan
    File: datasets.py
    Description: File containing dataset classes and methods to manage BIWI and AFLW2000
    Date: 15/05/21

    ** THE FOLLOWING CODE CONTAINS SECTIONS ADAPTED FROM THE TENSORFLOW VERSION OF THE HOPENET HEAD POSE ESTIMATION MODEL **
    ** URL: https://github.com/Oreobird/tf-keras-deep-head-pose **

    @InProceedings{Ruiz_2018_CVPR_Workshops,
    author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
    title = {Fine-Grained Head Pose Estimation Without Keypoints},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2018}
    }
"""

import os
import random

import cv2
import dlib
import numpy as np
import scipy.io as sio


# Method to split samples into test and training data
def split_samples(samples_file, train_file, test_file, ratio=0.8):
    with open(samples_file) as samples_fp:  # opening filename_list.txt
        lines = samples_fp.readlines()  # reading each line in the file
        random.shuffle(lines)  # shuffling lines in the file

        train_num = int(len(lines) * ratio)  # setting train number to the number of lines times the train ratio
        test_num = len(lines) - train_num  # setting test number to the remaining lines
        count = 0
        data = []

        for line in lines:  # iterating over every line
            count += 1
            data.append(line)
            if count == train_num:  # if count has reached the number of train lines
                with open(train_file, "w+") as train_fp:  # open the training file to write in
                    for d in data:
                        train_fp.write(d)  # writing the data
                data = []  # clearing the data

            if count == train_num + test_num:  # if count has reached the number of test lines
                with open(test_file, "w+") as test_fp:  # open the testing file to write in
                    for d in data:
                        test_fp.write(d)  # writing the data
                data = []  # clearing the data

    return train_num, test_num


# Method to creat a list from the filenames.txt file in each dataset
def get_list_from_filenames(file_path):
    with open(file_path) as f:  # opening filenames_list.txt
        lines = f.read().splitlines()  # creating a list of each line in the file
    return lines  # returning the list


# Class to represent the BIWI dataset
class Biwi:

    # Constructor
    def __init__(self, data_dir, data_file, batch_size=64, input_size=64, ratio=0.8):
        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.train_file = None
        self.test_file = None
        self.__gen_filename_list(self.data_dir + self.data_file)
        self.train_num, self.test_num = self.__gen_train_test_file(os.path.join(self.data_dir, 'train.txt'),
                                                                   os.path.join(self.data_dir, 'test.txt'), ratio=ratio)

    # Method to get input image from dataset
    def __get_input_img(self, file_name):
        detector = dlib.get_frontal_face_detector()

        img_ext = ".png"
        img_path = file_name + '_rgb' + img_ext
        img = dlib.load_rgb_image(img_path)

        faces = detector(img)

        crop_img = img[100:200, 100:200]
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            crop_img = img[y1 - 30:y2 + 30, x1 - 30:x2 + 30]

        crop_img = cv2.resize(src=crop_img, dsize=(self.input_size, self.input_size))

        crop_img = np.asarray(crop_img)
        normed_img = (crop_img - crop_img.mean()) / crop_img.std()

        return normed_img

    # Method to get correct training labels from the relevant image
    def __get_input_label(self, data_dir, file_name, annot_ext='.txt'):
        # Load pose in degrees
        pose_path = os.path.join(data_dir, file_name + '_pose' + annot_ext)
        pose_annot = open(pose_path, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)

        R = np.array(R)
        T = R[3, :]
        R = R[:3, :]
        pose_annot.close()

        R = np.transpose(R)

        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        # Bin values
        bins = np.array(range(-99, 99, 3))
        binned_labels = np.digitize([yaw, pitch, roll], bins) - 1

        cont_labels = [yaw, pitch, roll]

        return binned_labels, cont_labels

    # Method to get correct training labels from the relevant image
    def __gen_filename_list(self, filename_list_file):
        if not os.path.exists(filename_list_file):
            with open(filename_list_file, 'w+') as tlf:
                for root, dirs, files in os.walk(self.data_dir):
                    for subdir in dirs:
                        subfiles = os.listdir(os.path.join(self.data_dir, subdir))

                        for f in subfiles:
                            if os.path.splitext(f)[1] == '.png':
                                token = os.path.splitext(f)[0].split('_')
                                filename = token[0] + '_' + token[1]
                                # print(filename)
                                tlf.write(subdir + '/' + filename + '\n')

    # Method to create a file of test data and a file of training data
    def __gen_train_test_file(self, train_file, test_file, ratio=0.8):
        self.train_file = train_file
        self.test_file = test_file
        return split_samples(self.data_dir + self.data_file, self.train_file, self.test_file, ratio=ratio)

    # Method to generate objects to be used for training or testing
    def data_generator(self, shuffle=True, test=False):
        sample_file = self.train_file
        if test:
            sample_file = self.test_file

        filenames = get_list_from_filenames(sample_file)
        file_num = len(filenames)

        while True:

            if shuffle and not test:
                idx = np.random.permutation(range(file_num))
                filenames = np.array(filenames)[idx]

            max_num = file_num - (file_num % self.batch_size)

            for i in range(0, max_num, self.batch_size):
                batch_x = []
                batch_yaw = []
                batch_pitch = []
                batch_roll = []
                names = []

                for j in range(self.batch_size):
                    img = self.__get_input_img(filenames[i + j])
                    bin_labels, cont_labels = self.__get_input_label(self.data_dir, filenames[i + j])
                    batch_x.append(img)
                    batch_yaw.append([bin_labels[0], cont_labels[0]])
                    batch_pitch.append([bin_labels[1], cont_labels[1]])
                    batch_roll.append([bin_labels[2], cont_labels[2]])
                    names.append(filenames[i + j])

                batch_x = np.array(batch_x, dtype=np.float32)
                batch_yaw = np.array(batch_yaw)
                batch_pitch = np.array(batch_pitch)
                batch_roll = np.array(batch_roll)

                if test:
                    yield (batch_x, [batch_yaw, batch_pitch, batch_roll], names)
                else:
                    yield (batch_x, [batch_yaw, batch_pitch, batch_roll])
            if test:
                break


class AFLW2000:

    # Constructor
    def __init__(self, data_dir, filename_list, batch_size=16, input_size=64):
        self.data_dir = data_dir
        self.data_file = filename_list
        self.batch_size = batch_size
        self.input_size = input_size
        self.train_file = None
        self.test_file = None
        self.train_num, self.test_num = self.__gen_train_test_file(self.data_dir + 'train.txt',
                                                                   self.data_dir + 'test.txt')

    # Method to get continuous pose labels
    def __get_ypr_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pre_pose_params = mat['Pose_Para'][0]
        pose_params = pre_pose_params[:3]
        return pose_params  # pose params is an array containing the pitch and yaw of an image in radians

    # Method to get 2D landmarks from dataset
    def __get_pt2d_from_mat(self, mat_path):
        mat = sio.loadmat(mat_path)
        pt2d = mat['pt2d']
        return pt2d

    # Method to get correct training labels from the relevant image
    def __get_input_label(self, data_dir, file_name, annot_ext='.mat'):
        # Get the pose in radians
        pose = self.__get_ypr_from_mat(os.path.join(data_dir, file_name + annot_ext))

        # And convert to degrees.
        pitch = pose[0] * 180.0 / np.pi
        yaw = pose[1] * 180.0 / np.pi
        roll = pose[2] * 180.0 / np.pi

        cont_labels = [yaw, pitch, roll]  # setting continuous labels for mse regression, these are the degree vals

        # Bin values
        bins = np.array(range(-99, 99, 3))
        bin_labels = np.digitize([yaw, pitch, roll],
                                 bins) - 1  # setting the binned labels for classification, 0-2 deg = 33 bin, 3 degree per bin, rounded down

        return bin_labels, cont_labels

    # Method to create a file of test data and a file of training data
    def __gen_train_test_file(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        return split_samples(self.data_dir + self.data_file, self.train_file, self.test_file,
                             ratio=0.8)  # calling split samples method and passing values

    def train_num(self):
        return self.train_num

    def test_num(self):
        return self.test_num

    # Method to get input image from dataset
    def __get_input_img(self, data_dir, file_name, img_ext='.jpg', annot_ext='.mat'):
        img = cv2.imread(file_name + img_ext)  # data_dir +
        pt2d = self.__get_pt2d_from_mat(file_name + annot_ext)  # data_dir +

        # Crop the face loosely
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        Lx = abs(x_max - x_min)
        Ly = abs(y_max - y_min)
        Lmax = max(Lx, Ly) * 1.5
        center_x = x_min + Lx // 2
        center_y = y_min + Ly // 2

        x_min = center_x - Lmax // 2
        x_max = center_x + Lmax // 2
        y_min = center_y - Lmax // 2
        y_max = center_y + Lmax // 2

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        if x_min < 0:
            y_max -= abs(x_min)
            x_min = 0
        if y_min < 0:
            x_max -= abs(y_min)
            y_min = 0
        if x_max > img.shape[1]:
            y_min += abs(x_max - img.shape[1])
            x_max = img.shape[1]
        if y_max > img.shape[0]:
            x_min += abs(y_max - img.shape[0])
            y_max = img.shape[0]

        crop_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        crop_img = np.asarray(cv2.resize(crop_img, (self.input_size, self.input_size)))
        normed_img = (crop_img - crop_img.mean()) / crop_img.std()
        return normed_img

    # Method to generate objects to be used for training or testing
    def data_generator(self, shuffle=True):
        filenames = get_list_from_filenames(self.train_file)

        while True:
            if shuffle:
                idx = np.random.permutation(self.train_num)
                filenames = np.array(filenames)[idx]
            max_num = self.train_num - (self.train_num % self.batch_size)
            for i in range(0, max_num, self.batch_size):
                batch_img = []
                batch_yaw = []
                batch_pitch = []
                batch_roll = []
                names = []
                for j in range(self.batch_size):
                    img = self.__get_input_img(self.data_dir, filenames[i + j])
                    bin_labels, cont_labels = self.__get_input_label(self.data_dir, filenames[i + j])

                    batch_img.append(img)
                    batch_pitch.append([bin_labels[0], cont_labels[0]])
                    batch_yaw.append([bin_labels[1], cont_labels[1]])
                    batch_roll.append([bin_labels[2], cont_labels[2]])
                    names.append(filenames[i + j])

                batch_img = np.array(batch_img, dtype=np.float32)
                batch_yaw = np.array(batch_yaw)
                batch_pitch = np.array(batch_pitch)
                batch_roll = np.array(batch_roll)

                yield (batch_img, [batch_yaw, batch_pitch, batch_roll])
