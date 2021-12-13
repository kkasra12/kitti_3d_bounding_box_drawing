"""
$ tree -ld
├── data_object_calib
│    ├── testing
│    │   └── calib
│    └── training
│        └── calib
├── data_object_image_2
│    ├── testing
│    │   └── image_2
│    └── training
│        └── image_2
└── training
    └── label_2

$ du -hcL ./*
60M	    ./data_object_calib
30M	    ./data_object_calib/testing/calib
30M	    ./data_object_calib/training/calib
12G	    ./data_object_image_2
6.0G    ./data_object_image_2/testing/image_2
5.9G    ./data_object_image_2/training/image_2
30M	    ./training
30M	    ./training/label_2
12G	total
"""
import os
from os import listdir
from os.path import join, isfile
from typing import List, Generator, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

IMG_KEY = 'img'

DATA_FOLDER = "data"
CALIBRATION_FOLDER = "./data/data_object_calib/training/calib"
IMAGE_FOLDER = "./data/data_object_image_2/training/image_2"
LABEL_FOLDER = "./data/training/label_2"


def identity(x):
    return x


def read_one_calibration_file(name: str, parent_folder: str) -> pd.Series:
    t = pd.read_csv(join(parent_folder, name), delimiter=" ", header=None)
    t.iloc[:, 0] = t[0].apply(lambda x: x[:-1])
    t = t.set_index(0)
    t_stack = t.stack()
    t_stack.name = name.split('.')[0]
    return t_stack


def read_calibration(address: str = None, progress_bar: str = "console") -> pd.DataFrame:
    """
    this function returns all calibrations
    index P0_0
    :param progress_bar: can be console or notebook or None
    :param address:
    :return:
    """
    if address is None:
        address = CALIBRATION_FOLDER
    progress_bar = return_progressbar(progress_bar)
    # TODO: sort indexes
    return pd.concat([read_one_calibration_file(file_name, address) for file_name in progress_bar(listdir(address))],
                     axis=1).T


def return_progressbar(progress_bar):
    progress_bar_options = ['console', 'notebook', None, False, "None"]
    if progress_bar == progress_bar_options[0]:
        progress_bar = tqdm
    elif progress_bar == progress_bar_options[1]:
        progress_bar = tqdm_notebook
    elif progress_bar in progress_bar_options[2:]:
        progress_bar = identity
    else:
        raise ValueError("progress bar must be on of these:" + "\n\t".join(progress_bar_options))
    return progress_bar


def read_labels(address: str = None, progress_bar="console"):
    """
    All values (numerical or strings) are separated via spaces,
    each row corresponds to one object. The 15 columns represent:

    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.

    Here, 'DontCare' labels denote regions in which objects have not been labeled,
    for example because they have been too far away from the laser scanner. To
    prevent such objects from being counted as false positives our evaluation
    script will ignore objects detected in don't care regions of the test set.
    You can use the don't care labels in the training set to avoid that your object
    detector is harvesting hard negatives from those areas, in case you consider
    non-object regions from the training images as negative examples.

    The coordinates in the camera coordinate system can be projected in the image
    by using the 3x4 projection matrix in the calib folder, where for the left
    color camera for which the images are provided, P2 must be used. The
    difference between rotation_y and alpha is, that rotation_y is directly
    given in camera coordinates, while alpha also considers the vector from the
    camera center to the object center, to compute the relative orientation of
    the object with respect to the camera. For example, a car which is facing
    along the X-axis of the camera coordinate system corresponds to rotation_y=0,
    no matter where it is located in the X/Z plane (bird's eye view), while
    alpha is zero only, when this object is located along the Z-axis of the
    camera. When moving the car away from the Z-axis, the observation angle
    will change.

    To project a point from Velodyne coordinates into the left color image,
    you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y
    For the right color image: x = P3 * R0_rect * Tr_velo_to_cam * y

    Note: All matrices are stored row-major, i.e., the first values correspond
    to the first row. R0_rect contains a 3x3 matrix which you need to extend to
    a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
    Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
    in the same way!

    Note, that while all this information is available for the training data,
    only the data which is actually needed for the particular benchmark must
    be provided to the evaluation server. However, all 15 values must be provided
    at all times, with the unused ones set to their default values (=invalid) as
    specified in writeLabels.m. Additionally a 16'th value must be provided
    with a floating value of the score for a particular detection, where higher
    indicates higher confidence in the detection. The range of your scores will
    be automatically determined by our evaluation server, you don't have to
    normalize it, but it should be roughly linear. If you use writeLabels.m for
    writing your results, this function will take care of storing all required
    data correctly.

    :param address:
    :param progress_bar: can be console or notebook or None
    :return:
    """
    if address is None:
        address = LABEL_FOLDER
    progress_bar = return_progressbar(progress_bar)
    columns = [('type', 0),
               ('truncated', 0),
               ('occluded', 0),
               ('alpha', 0),
               ('bbox', 0),
               ('bbox', 1),
               ('bbox', 2),
               ('bbox', 3),
               ('dimensions', 0),
               ('dimensions', 1),
               ('dimensions', 2),
               ('location', 0),
               ('location', 1),
               ('location', 2),
               ('rotation_y', 0),
               ('score', 0),
               ('filename', 0)]
    columns = pd.MultiIndex.from_tuples(columns, names=[1, 2])
    all_dfs: List[pd.DataFrame] = []
    for filename in progress_bar(os.listdir(address)):
        all_dfs.append(pd.read_csv(join(address, filename), delimiter=" ", names=columns))
        all_dfs[-1]['filename'] = filename.split(".")[0]
    return pd.concat(all_dfs).reset_index(drop=True)
    # TODO: convert ['filename'][0] into numbers and set for index
    # NOTE: you may need to have multi indexing(each file has several entries)


def read_image(name, parent_folder=None, file_name_len=10):
    if parent_folder is None:
        parent_folder = IMAGE_FOLDER
    if "." not in name:
        name = f"{name:0>{file_name_len - 4}}.png"
    elif name.split(".")[-1].lower() != "png":
        raise ValueError("file extension must be png")
    img = cv2.imread(join(parent_folder, name))
    if img is None:
        raise ValueError(f"file {name} not found or inaccessible")
    return img


def read_all_images(parent_folder=None) -> Generator[Tuple[str, np.ndarray], None, None]:
    if parent_folder is None:
        parent_folder = IMAGE_FOLDER
    for img in listdir(parent_folder):
        if img.split(".")[-1] == "png":
            yield img, cv2.imread(join(parent_folder, img))


def write_to_hdf(filename,
                 calibration_dataset: pd.DataFrame = None,
                 label_dataset: pd.DataFrame = None,
                 image_folder: str = None,
                 calibration_key='calibration',
                 label_key='label',
                 image_key=IMG_KEY,
                 verbose=1,
                 overwrite=False,
                 progress_bar='console'):
    """

    :param filename:
    :param calibration_dataset: if false it wont store calibration dataset
    :param label_dataset: if false it wont store label dataset
    :param image_folder: if false it wont store images
    :param calibration_key:
    :param label_key:
    :param image_key:
    :param verbose:
    :param overwrite: if overwrite is false and the is another file with same file the code will add numeric suffix to
                        create a new file
    :param progress_bar:
    :return:
    """
    # TODO: add compression level
    if calibration_key == label_key:
        raise ValueError("both keys cant be same")

    num = 0
    filename_ = filename
    while isfile(filename_) and not overwrite:
        filename_ = f"{filename.rsplit('.', 1)[0]}_{num}.{filename.rsplit('.', 1)[1]}"
        num += 1
    filename = filename_
    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    if calibration_dataset is None:
        if verbose:
            print("reading calibration dataset from files...")
        calibration_dataset = read_calibration(progress_bar=progress_bar)
    if calibration_dataset is not False:
        calibration_dataset.to_hdf(filename, key=calibration_key, mode=mode)
        if verbose:
            print(f"calibration database is saved to {filename} with key:{calibration_key}")

    if label_dataset is None:
        if verbose:
            print("reading labels dataset from files...")
        label_dataset = read_labels(progress_bar=progress_bar)
    if label_dataset is not False:
        label_dataset.to_hdf(filename, key=label_key, mode='a')
        if verbose:
            print(f"label database is saved to {filename} with key:{label_key}")

    if image_folder is None:
        image_folder = IMAGE_FOLDER
        if verbose:
            print("reading images...")
        progress_bar = return_progressbar(progress_bar)
        with h5py.File(filename, 'a') as f:
            for img_name, img in progress_bar(read_all_images(parent_folder=image_folder),
                                              total=len(listdir(image_folder))):
                f.create_dataset(f"{image_key}/{img_name.split('.')[0]}", data=img)
                # TODO: add data type while saving hdf5 files
        print(f"image database is saved to {filename} with key:{image_key}")


def read_img_from_hdf(hdf_file, img_number, img_key=IMG_KEY) -> np.ndarray:
    with h5py.File(hdf_file) as f:
        return np.array(f[img_key][img_number])


if __name__ == '__main__':
    calibrations = read_calibration()
