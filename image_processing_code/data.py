"""
    Name: Ross Monaghan
    File: data.py
    Description: File containing methods to manage data input for saliency mapping
    Date: 15/05/21

    ** THE FOLLOWING CODE IS TAKEN FROM THE MSI-NET GITHUB REPOSITORY **

    ** URL: https://github.com/alexanderkroner/saliency **

    @article{kroner2020contextual,
      title={Contextual encoder-decoder network for visual saliency prediction},
      author={Kroner, Alexander and Senden, Mario and Driessens, Kurt and Goebel, Rainer},
      url={http://www.sciencedirect.com/science/article/pii/S0893608020301660},
      doi={https://doi.org/10.1016/j.neunet.2020.05.004},
      journal={Neural Networks},
      publisher={Elsevier},
      year={2020},
      volume={129},
      pages={261--270},
      issn={0893-6080}
    }

"""

import os
import numpy as np
import tensorflow as tf


class TEST:
    """This class represents test set instances used for inference through
       a trained network. All stimuli are resized to the preferred spatial
       dimensions of the chosen model. This can, however, lead to cases of
       excessive image padding.

    Returns:
        object: A dataset object that holds all test set instances
                specified under the path variable.
    """

    def __init__(self, data_path):
        self._target_size = (240, 320)
        self._dir_stimuli_test = data_path

    def load_data(self):
        test_list_x = _get_file_list(self._dir_stimuli_test)
        test_set = _fetch_dataset(files=test_list_x, target_size=self._target_size)

        return test_set


def get_dataset_iterator(data_path):
    """Entry point to make an initializable dataset iterator for either
       training or testing a model by calling the respective dataset class.

    Args:
        phase (str): Holds the current phase, which can be "train" or "test".
        dataset (str): Denotes the dataset to be used during training or the
                       suitable resizing procedure when testing a model.
        data_path (str): Points to the directory where training or testing
                         data instances are stored.

    Returns:
        iterator: An initializable dataset iterator holding the relevant data.
        initializer: An operation required to initialize the correct iterator.
    """

    test_class = TEST(data_path)
    test_set = test_class.load_data()

    iterator = tf.compat.v1.data.Iterator.from_structure(test_set.output_types, test_set.output_shapes)
    init_op = iterator.make_initializer(test_set)
    next_element = iterator.get_next()

    return next_element, init_op


def postprocess_saliency_map(saliency_map, target_size):
    """This function resizes and crops a single saliency map to the original
       dimensions of the input image. The output is then encoded as a jpeg
       file suitable for saving to disk.

    Args:
        saliency_map (tensor, float32): 3D tensor that holds the values of a
                                        saliency map in the range from 0 to 1.
        target_size (tensor, int32): 1D tensor that specifies the size to which
                                     the saliency map is resized and cropped.

    Returns:
        tensor, str: A tensor of the saliency map encoded as a jpeg file.
    """

    saliency_map *= 255.0
    saliency_map = _resize_image(saliency_map, target_size, True)
    saliency_map = _crop_image(saliency_map, target_size)
    saliency_map = tf.compat.v1.round(saliency_map)
    saliency_map = tf.compat.v1.cast(saliency_map, tf.compat.v1.uint8)

    saliency_map_jpeg = tf.compat.v1.image.encode_jpeg(saliency_map, "grayscale", 100)

    return saliency_map_jpeg


def _fetch_dataset(files, target_size):
    """Here the list of file directories is shuffled (only when training),
       loaded, batched, and prefetched to ensure high GPU utilization.

    Args:
        files (list, str): A list that holds the paths to all file instances.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be reshaped.
        shuffle (bool): Determines whether the dataset will be shuffled or not.
        online (bool, optional): Flag that decides whether the batch size must
                                 be 1 or can take any value. Defaults to False.

    Returns:
        object: A dataset object that contains the batched and prefetched data
                instances along with their shapes and file paths.
    """

    dataset = tf.compat.v1.data.Dataset.from_tensor_slices(files)

    map_func = lambda *files: _parse_function(files, target_size)
    num_parallel_calls = tf.data.experimental.AUTOTUNE

    dataset = dataset.map(map_func=map_func,  num_parallel_calls=num_parallel_calls)

    batch_size = 1

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(5)

    return dataset


def _parse_function(files, target_size):
    """This function reads image data dependent on the image type and
       whether it constitutes a stimulus or saliency map. All instances
       are then reshaped and padded to yield the target dimensionality.

    Args:
        files (tuple, str): A tuple with the paths to all file instances.
                            The first element contains the stimuli and, if
                            present, the second one the ground truth maps.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be reshaped.

    Returns:
        list: A list that holds the image instances along with their
              shapes and file paths.
    """

    image_list = []

    for count, filename in enumerate(files):
        image_str = tf.compat.v1.read_file(filename)
        channels = 3 if count == 0 else 1

        image = tf.compat.v1.cond(tf.compat.v1.image.is_jpeg(image_str),
                                  lambda: tf.compat.v1.image.decode_jpeg(image_str, channels=channels),
                                  lambda: tf.compat.v1.image.decode_png(image_str, channels=channels))
        original_size = tf.compat.v1.shape(image)[:2]

        image = _resize_image(image, target_size)
        image = _pad_image(image, target_size)

        image_list.append(image)

    image_list.append(original_size)
    image_list.append(files)

    return image_list


def _resize_image(image, target_size, overfull=False):
    """This resizing procedure preserves the original aspect ratio and might be
       followed by padding or cropping. Depending on whether the target size is
       smaller or larger than the current image size, the area or bicubic
       interpolation method will be utilized.

    Args:
        image (tensor, uint8): A tensor with the values of an image instance.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be resized.
        overfull (bool, optional): Denotes whether the resulting image will be
                                   larger or equal to the specified target
                                   size. This is crucial for the following
                                   padding or cropping. Defaults to False.

    Returns:
        tensor, float32: 4D tensor that holds the values of the resized image.

    .. seealso:: The reasoning for using either area or bicubic interpolation
                 methods is based on the OpenCV documentation recommendations.
                 [https://bit.ly/2XAavw0]
    """

    current_size = tf.compat.v1.shape(image)[:2]

    height_ratio = target_size[0] / current_size[0]
    width_ratio = target_size[1] / current_size[1]

    if overfull:
        target_ratio = tf.compat.v1.maximum(height_ratio, width_ratio)
    else:
        target_ratio = tf.compat.v1.minimum(height_ratio, width_ratio)

    target_size = tf.compat.v1.cast(current_size, tf.compat.v1.float64) * target_ratio
    target_size = tf.compat.v1.cast(tf.compat.v1.round(target_size), tf.compat.v1.int32)

    shrinking = tf.compat.v1.cond(
        tf.compat.v1.logical_or(current_size[0] > target_size[0],
                            current_size[1] > target_size[1]),
                            lambda: tf.compat.v1.constant(True),
                            lambda: tf.compat.v1.constant(False))

    image = tf.compat.v1.expand_dims(image, 0)
    image = tf.compat.v1.cond(shrinking,
                          lambda: tf.compat.v1.compat.v1.image.resize_area(image, target_size, align_corners=True),
                          lambda: tf.compat.v1.compat.v1.image.resize_bicubic(image, target_size, align_corners=True))

    image = tf.compat.v1.clip_by_value(image[0], 0.0, 255.0)

    return image


def _pad_image(image, target_size):
    """A single image, either stimulus or saliency map, will be padded
       symmetrically with the constant value 126 or 0 respectively.

    Args:
        image (tensor, float32): 3D tensor with the values of the image data.
        target_size (tuple, int): A tuple that specifies the size to which
                                  the data will be resized.

    Returns:
        tensor, float32: 3D tensor that holds the values of the padded image.
    """

    current_size = tf.compat.v1.shape(image)

    pad_constant_value = tf.compat.v1.cond(tf.compat.v1.equal(current_size[2], 3),
                                           lambda: tf.compat.v1.constant(126.0),
                                           lambda: tf.compat.v1.constant(0.0))

    pad_vertical = (target_size[0] - current_size[0]) / 2
    pad_horizontal = (target_size[1] - current_size[1]) / 2

    pad_top = tf.compat.v1.floor(pad_vertical)
    pad_bottom = tf.compat.v1.ceil(pad_vertical)
    pad_left = tf.compat.v1.floor(pad_horizontal)
    pad_right = tf.compat.v1.ceil(pad_horizontal)

    padding = [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    image = tf.compat.v1.pad(image, padding, constant_values=pad_constant_value)

    return image


def _crop_image(image, target_size):
    """A single saliency map will be cropped according the specified target
       size by extracting the central region of the image and correctly
       removing the added padding.

    Args:
        image (tensor, float32): 3D tensor with the values of a saliency map.
        target_size (tensor, int32): 2D tensor that specifies the size to
                                     which the data will be cropped.

    Returns:
        tensor, float32: 3D tensor that holds the values of the saliency map
                         with cropped dimensionality.
    """

    current_size = tf.compat.v1.shape(image)[:2]

    crop_vertical = (current_size[0] - target_size[0]) / 2
    crop_horizontal = (current_size[1] - target_size[1]) / 2

    crop_top = tf.compat.v1.cast(tf.compat.v1.floor(crop_vertical), tf.compat.v1.int32)
    crop_left = tf.compat.v1.cast(tf.compat.v1.floor(crop_horizontal), tf.compat.v1.int32)

    border_bottom = crop_top + target_size[0]
    border_right = crop_left + target_size[1]

    image = image[crop_top:border_bottom, crop_left:border_right, :]

    return image


def _get_file_list(data_path):
    """This function detects all image files within the specified parent
       directory for either training or testing. The path content cannot
       be empty, otherwise an error occurs.

    Args:
        data_path (str): Points to the directory where training or testing
                         data instances are stored.

    Returns:
        list, str: A sorted list that holds the paths to all file instances.
    """

    data_list = []

    if os.path.isfile(data_path):
        data_list.append(data_path)
    else:
        for subdir, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    data_list.append(os.path.join(subdir, file))

    data_list.sort()

    if not data_list:
        raise FileNotFoundError("No data was found")

    return data_list


def _get_random_indices(list_length):
    """A helper function to generate an array of randomly shuffled indices
       to divide the MIT1003 and CAT2000 datasets into training and validation
       instances.

    Args:
        list_length (int): The number of indices that is randomly shuffled.

    Returns:
        array, int: A 1D array that contains the shuffled data indices.
    """

    indices = np.arange(list_length)
    prng = np.random.RandomState(42)
    prng.shuffle(indices)

    return indices


def _check_consistency(zipped_file_lists, n_total_files):
    """A consistency check that makes sure all files could successfully be
       found and stimuli names correspond to the ones of ground truth maps.

    Args:
        zipped_file_lists (tuple, str): A tuple of train and valid path names.
        n_total_files (int): The total number of files expected in the list.
    """

    assert len(list(zipped_file_lists)) == n_total_files, "Files are missing"

    for file_tuple in zipped_file_lists:
        file_names = [os.path.basename(entry) for entry in list(file_tuple)]
        file_names = [os.path.splitext(entry)[0] for entry in file_names]
        file_names = [entry.replace("_fixMap", "") for entry in file_names]
        file_names = [entry.replace("_fixPts", "") for entry in file_names]

        assert len(set(file_names)) == 1, "File name mismatch"
