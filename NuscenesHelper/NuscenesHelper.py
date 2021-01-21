from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
import numpy as np
from scipy import spatial


class NuscenesHelper:
    def __init__(self, version: str = 'v1.0-mini', dataroot: str = '/data/sets/nuscenes'):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    def filter_occluded_objects(self, saving_file_name, sensor='CAM_FRONT', report_file_path=None):
        txt_file = open(saving_file_name, "w+")
        # report_file = open(report_file_path,"w+")
        # report_file.write("SENSOR="+sensor+"\n")
        scenes = self.nusc.scene
        for scene in scenes:
            print("processing scene ", scene['token'])
            # report_file.write("scene "+scene['token']+"\n")
            occluded_tokens = self.get_occluded_annos_in_scence(scene, sensor)
            for token in occluded_tokens:
                txt_file.write(token + "\n")
        txt_file.close()

    def get_occluded_annos_in_scence(self, scene, sensor):
        occluded_ann_tokens = []
        samples = self.get_samples(scene)
        for sample in samples:
            occluded_anno_tokens_in_sample = self.get_occluded_ann_tokens_in_sample(sample, sensor)
            occluded_ann_tokens += occluded_anno_tokens_in_sample
            print("----There are ", len(occluded_anno_tokens_in_sample), " in sample ", sample['token'], "of scene",
                  scene['token'])
        return occluded_ann_tokens

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

    def get_occluded_ann_tokens_in_sample(self, sample, sensor):
        occluded_ann_tokens = []
        anns = self.get_annos(sample)
        camera_data = self.nusc.get('sample_data', sample['data'][sensor])
        for ann in anns:
            if self.is_occluded(ann, anns, camera_data):
                occluded_ann_tokens.append(ann['token'])
        return occluded_ann_tokens

    def get_annos(self, sample):
        anns = []
        ann_tokens = sample["anns"]
        for ann_token in ann_tokens:
            ann = self.nusc.get("sample_annotation", ann_token)
            anns.append(ann)
        return anns

    """
        check if an `anno` is occluded by another in `anns`
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

        corners = view_points(box.corners(), camera_intrinsic_matrix, normalize=True)[:2, :]
        # two_d_bb1 = NuscenesHelper.min_max_x_y(corners)
        for annotation in anns:
            _, boxes, other_camera_intrinsic_matrix = self.nusc.get_sample_data(camera_data['token'],
                                                                                selected_anntokens=[
                                                                                    annotation['token']])
            if len(boxes) != 1:
                continue
            other_box = boxes[0]
            if box.center[2] < other_box.center[2]:
                continue
            other_corners = view_points(other_box.corners(), other_camera_intrinsic_matrix, normalize=True)[:2, :]
            # two_d_bb2 = NuscenesHelper.min_max_x_y(other_corners)
            if NuscenesHelper.is_occluded_2D(corners, other_corners):
                return True
        return False

    def get_scenes(self):
        return self.nusc.scene

    def make_report(self, sensor, scene, occluded_ann_tokens):
        lines = []
        lines.append()

    @staticmethod
    def is_occluded_2D(points1, points2):
        convex_hull = spatial.ConvexHull(np.concatenate((points1.T, points2.T)))
        if np.amin(convex_hull.vertices) > 7:
            return True
        return False
