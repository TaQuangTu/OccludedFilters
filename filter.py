from NuscenesHelper.NuscenesHelper import NuscenesHelper

SENSORS = ["CAM_FRONT", 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
if __name__ == "__main__":
    occluded_ann_tokens_file = "occluded_object_tokens.txt"
    report_file = "report.txt"
    datasetHelper = NuscenesHelper(version='v1.0-mini', dataroot='data/set/nuscenes/v1.0-mini')

    for sensor in SENSORS:  # filter occluded objects for camera sensors
        datasetHelper.filter_occluded_objects(sensor + occluded_ann_tokens_file, sensor, "reports/" + sensor + "_" + report_file)
