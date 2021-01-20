from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
import numpy as np

class NuscenesHelper:
    def __init__(self, version: str = 'v1.0-mini', dataroot: str = '/data/sets/nuscenes'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    def filter_occluded_objects(self, saving_file_name):
        txt_file = open(saving_file_name, "w+")
        scenes = self.nusc.scene
        for scene in scenes:
            print("processing scene ",scene['token'])
            occluded_tokens = self.get_occluded_objects_in_scence(scene)
            for token in occluded_tokens:
                txt_file.write(token + "\n")
        txt_file.close()

    def get_occluded_objects_in_scence(self, scene):
        occluded_object_tokens = []
        samples = self.get_samples(scene)
        for sample in samples:
            occluded_object_tokens_in_sample = self.get_occluded_object_tokens_in_sample(sample)
            occluded_object_tokens += occluded_object_tokens_in_sample
            print("----There are ",len(occluded_object_tokens_in_sample)," in sample ",sample['token'],"of scene",scene['token'])
        return occluded_object_tokens

    def get_samples(self, scene):
        samples = []
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        sample = first_sample
        if sample is not None:
            samples.append(sample)
        while sample["next"] != "":
            sample = self.nusc.get("sample", sample["next"])
            samples.append(sample)
        return samples

    '''
        TODO: take care the sensor parameter
    '''

    def get_occluded_object_tokens_in_sample(self, sample, sensor="CAM_FRONT"):
        occluded_object_tokens = []
        objects = self.get_objects(sample)
        camera_data = self.nusc.get('sample_data', sample['data'][sensor])
        for object in objects:
            if self.is_occluded(object, objects, camera_data):
                occluded_object_tokens.append(object['token'])
        return occluded_object_tokens

    def get_objects(self, sample):
        objects = []
        ann_tokens = sample["anns"]
        for ann_token in ann_tokens:
            object = self.nusc.get("sample_annotation", ann_token)
            objects.append(object)
        return objects

    """
        check if an `object` is occluded by another in `objects`
    """

    def is_occluded(self, anno, anns, camera_data):
        # Plot CAMERA view.
        sample_record = self.nusc.get('sample', anno['sample_token'])
        assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'
        cam_path, boxes, camera_intrinsic_matrix = self.nusc.get_sample_data(camera_data['token'],
                                                                             selected_anntokens=[anno['token']])

        if len(boxes) != 1:
            return True
        box = boxes[0]
        corners = view_points(box.corners(),camera_intrinsic_matrix,normalize=True)[:2, :]
        two_d_bb1 = NuscenesHelper.min_max_x_y(corners)
        for annotation in anns:
            _, boxes, camera_intrinsic_matrix = self.nusc.get_sample_data(camera_data['token'],
                                                                                 selected_anntokens=[annotation['token']])
            if len(boxes)!=1:
                continue
            other_box = boxes[0]
            if box.center[2] < other_box.center[2]:
                continue
            other_corners = view_points(other_box.corners(), camera_intrinsic_matrix, normalize=True)[:2, :]
            two_d_bb2 = NuscenesHelper.min_max_x_y(other_corners)
            if NuscenesHelper.check_occluded_2D(two_d_bb1, two_d_bb2):
                return True
        return False

    '''
    Return xmin, ymin, xmax, ymax from list of 2D points
    '''
    @staticmethod
    def min_max_x_y(corners):
        x_min_y_min = np.amin(corners,axis=1)
        x_max_y_max = np.amax(corners,axis=1)
        return x_min_y_min[0], x_min_y_min[1], x_max_y_max[0], x_max_y_max[1]
    '''
    check if 2D bb1 is occluded by 2D bb2
    '''
    @staticmethod
    def check_occluded_2D(two_d_bb1, two_d_bb2):
        if two_d_bb1[0]>two_d_bb2[0] and two_d_bb1[1] > two_d_bb2[1] and two_d_bb1[2]<two_d_bb2[2] and two_d_bb1[3] < two_d_bb2[3]:
            return True
        return False

    def get_scenes(self):
        return self.nusc.scene