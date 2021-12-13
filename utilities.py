from typing import Tuple, List, Union, Sequence, Any

import cv2
import numpy as np
import pandas as pd

from prepare_data import read_img_from_hdf


def camera_coordinates_to_image_plane(point: Tuple[float, float, float], calibration: pd.Series):
    P2 = calibration['P2'].values.reshape((3, 4))
    R_rect = np.identity(4)
    R_rect[:3, :3] = calibration['R0_rect'].values.reshape((3, 3))
    point = np.array([*point, 1])
    # T_cam_velo = np.vstack([calibration['Tr_velo_to_cam'].values.reshape((3, 4)),
    #                         [[0, 0, 0, 1]]])
    # T_velo_imu = np.vstack([calibration['Tr_imu_to_velo'].values.reshape((3, 4)),
    #                         [[0, 0, 0, 1]]])

    # print(calibration.index)
    # print(f"{point=}")
    # return P2 @ point
    y = P2 @ R_rect @ point
    y /= y[-1]
    return y[:2]


def rotate_yaw(t: float) -> np.ndarray:
    """

    :param t:
    :return:
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


class image:
    win_num = 0

    def __init__(self, img, win_name=None, calibration=None):
        if win_name is None:
            win_name = f'unnamed_{self.__class__.win_num}'
            self.__class__.win_num += 1
        self.img = img
        self.win_name = win_name
        self.labels = []
        self.cubes = []
        self.calibration = calibration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            cv2.imshow(self.win_name, self.img)
            cv2.waitKey()
            cv2.destroyWindow(self.win_name)

    def put_point(self, point, color=(255, 0, 0)):
        center = [int(i) for i in point[:2]]
        # print(f"{center=}")
        self.img = cv2.circle(img=self.img,
                              center=center,
                              radius=5,
                              thickness=-1,
                              color=color)

    def put_line(self, line: Union[List, np.ndarray], color=(255, 0, 0)):
        """
        draws a 2D line between two points
        line is a list of two points
        :param line: line[0] is the start point, line[1] is the end point
        :param color: color of the line
        :return:
        """
        if isinstance(line, list):
            line = np.array(line)
        if line.shape != (2, 2):
            raise ValueError(f"line should be (2,2) not {line.shape=}")
        self.img = cv2.line(img=self.img,
                            pt1=line[0],
                            pt2=line[1],
                            color=color,
                            thickness=1)

    def put_text(self, text, point, color=(255, 0, 0), color_background=(0, 0, 0), font_scale: float = 1):
        text_size = cv2.getTextSize(text,
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale,
                                    thickness=1)[0]
        self.img = cv2.rectangle(self.img,
                                 pt1=point,
                                 pt2=point + text_size * np.array([1, -1]),
                                 color=color_background,
                                 thickness=-1)
        self.img = cv2.putText(img=self.img,
                               text=text,
                               org=point,
                               fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=font_scale,
                               color=color,
                               thickness=2)

    def put_rect(self, rect: np.ndarray, color=(255, 0, 0)):
        if rect.shape != (4, 2):
            raise ValueError(f"rect should be (4,2) not {rect.shape=}")
        self.put_line(line=[rect[0], rect[1]], color=color)
        self.put_line(line=[rect[1], rect[2]], color=color)
        self.put_line(line=[rect[2], rect[3]], color=color)
        self.put_line(line=[rect[3], rect[0]], color=color)

    def put_cube(self, cube: np.ndarray, color=(255, 0, 0)):
        """

        :param cube: is a matrix of shape (8, 3), where each row is a point of the cube
        first row is the front bottom left, second row is the front bottom right, third row is the front top right,
        fourth row is the front top left, fifth row is the back bottom left, sixth row is the back bottom right,
        seventh row is the back top right, eighth row is the back top left
        :param color:
        :return:
        """
        cube = cube.copy()
        self.put_rect(rect=cube[:4], color=color)
        self.put_rect(rect=cube[4:], color=color)
        self.put_line(line=[cube[0], cube[4]], color=color)
        self.put_line(line=[cube[1], cube[5]], color=color)
        self.put_line(line=[cube[2], cube[6]], color=color)
        self.put_line(line=[cube[3], cube[7]], color=color)
        for point in cube[:, :]:
            self.put_point(point, color=(0, 255, 0))
            # print(f"{point=}")

    @staticmethod
    def create_cube(label, calibration) -> np.ndarray:
        """
        this function will find the cube point from labels
        :return:
        """
        # print(f"{label['dimensions'].values=}")
        l, w, h = label['dimensions'].values[::-1]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        cube = np.vstack([x, y, z])
        cube = rotate_yaw(label['rotation_y'].values[0]) @ cube
        cube = cube + label['location'].values[:, None]
        cube = np.apply_along_axis(lambda x: camera_coordinates_to_image_plane(x, calibration), axis=0, arr=cube).T
        cube = cube.astype(np.int32)
        # print(f"{cube=}")
        return cube

    def draw_cube_from_label(self, label, calibration=None, print_label=True, zmax=None, label_name=None):
        if calibration is None:
            calibration = self.calibration
        if label_name is None:
            label_name = label['type'].values[0]
            # TODO:
        cube = self.create_cube(label, calibration)
        self.put_cube(cube)
        z = label['location'].values[2]
        if zmax is None:
            font_scale = 1
        else:
            font_scale = max(0.3, 1 - z / zmax)
        if print_label:
            self.put_text(text=label_name, point=cube[0], color=(0, 0, 255),
                          font_scale=font_scale)
        self.cubes.append(cube)
        self.labels.append(label)

    @property
    def img_rgb(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)


def partly_annotate(fig_number: int,
                    car_instances: List[int] = None,
                    all_labels=None,
                    all_calibrations=None,
                    dataset_filename="dataset.hd5",
                    return_values: Sequence = None) -> Tuple[Union[image, Any], ...]:
    if return_values is not None:
        return_values = set(return_values)
        if len(return_values) == 0:
            raise ValueError("return_values cannot be empty")
        possible_return_values = {'image', 'labels', 'calibrations'}
        if return_values-possible_return_values:
            raise ValueError(f"return_values contains invalid values: {return_values-possible_return_values}")
    else:
        return_values = {'image', 'labels', 'calibrations'}

    if all_labels is None:
        all_labels = pd.read_hdf(dataset_filename, key="label")
    all_labels = all_labels.loc[(all_labels[('type', 0)] == 'Car'), :].drop(columns=[('type', 0)]).set_index(
        ('filename', 0)).sort_index()
    if all_calibrations is None:
        all_calibrations = pd.read_hdf(dataset_filename, key="calibration")
    image_file_name = f"{fig_number:0>6}"
    img = image(read_img_from_hdf(dataset_filename, image_file_name))
    image_calibration = all_calibrations.loc[image_file_name, :]
    this_file_labels = all_labels.loc[image_file_name, :]
    if car_instances is None:
        car_instances = list(range(this_file_labels.shape[0]))
    for car_instance in car_instances:
        label = this_file_labels.iloc[car_instance, :]
        img.draw_cube_from_label(label=label, calibration=image_calibration, label_name=f'Car{car_instance}')
    return_tuple = []
    if 'image' in return_values:
        return_tuple.append(img)
    if 'labels' in return_values:
        return_tuple.append(this_file_labels.iloc[car_instances, :])
    if 'calibrations' in return_values:
        return_tuple.append(image_calibration)
    return tuple(return_tuple)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    labels: pd.DataFrame = pd.read_hdf("dataset.hd5", key='label')
    # print(*list(zip(range(len(labels.columns)), labels.columns)), sep='\n')

    # noinspection PyTypeChecker
    calibrations: pd.DataFrame = pd.read_hdf("dataset.hd5", key="calibration")

    test_case_number = 10
    test_case_filename = f"{test_case_number:0>6}"
    test_case_labels = labels[labels[('filename', 0)] == test_case_filename]
    test_case_calibration = calibrations.loc[test_case_filename, :]
    #
    # point = camera_coordinates_to_image_plane(test_case_labels['location'].to_list(), test_case_calibration)
    with image(read_img_from_hdf('dataset.hd5', test_case_filename)) as img:
        for label in test_case_labels.iterrows():
            img.draw_cube_from_label(label[1], test_case_calibration, zmax=test_case_labels[('location', 2)].max())
