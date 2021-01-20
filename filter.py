from NuscenesHelper.NuscenesHelper import NuscenesHelper

if __name__ == "__main__":
    txt_file = "occluded_object_tokens.txt"
    datasetHelper = NuscenesHelper(version='v1.0-mini', dataroot='data/set/nuscenes/v1.0-mini')
    datasetHelper.filter_occluded_objects(txt_file)
