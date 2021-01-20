import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
from NuscenesHelper.NuscenesHelper import NuscenesHelper

CAM_FRONT = "CAM_FRONT"
if __name__ == "__main__":
    files = glob.glob("VisualizedImages/*")
    for file in files:
        os.remove(file)
    occluded_tokens = open("occluded_object_tokens.txt", "r").readlines()
    occluded_tokens = [x.replace("\n", "") for x in occluded_tokens]
    datasetHelper = NuscenesHelper(version='v1.0-mini', dataroot='data/set/nuscenes/v1.0-mini')
    scenes = datasetHelper.get_scenes()
    for scene in scenes:
        print("Processing scene", scene['token'])
        samples = datasetHelper.get_samples(scene)
        for sample in samples:
            print("------Processing sample", sample['token'])
            out_path = "VisualizedImages/" + sample['token'] + ".jpg"
            annotations = sample['anns']
            visible_anno_tokens = [x for x in annotations if x not in occluded_tokens]
            fig, axes = plt.subplots(1, 3, figsize=(27, 9))

            # show origin image at left
            data_path = datasetHelper.nusc.get_sample_data_path(sample['data'][CAM_FRONT])
            origin_image = Image.open(data_path)
            axes[0].imshow(origin_image)
            axes[0].set_title("origin")
            axes[0].axis('off')
            axes[0].set_aspect('equal')
            # show origin image at center along with all annotations
            data_path, boxes, camera_intrinsic = datasetHelper.nusc.get_sample_data(sample['data'][CAM_FRONT])
            im = Image.open(data_path)
            axes[1].imshow(im)
            axes[1].set_title("All annos("+str(len(boxes))+")")
            axes[1].axis('off')
            axes[1].set_aspect('equal')

            for box in boxes:
                c = np.array(datasetHelper.nusc.colormap[box.name]) / 255.0
                box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # show origin image at center along with all annotations EXCEPT occluded annotations
            data_path, boxes, camera_intrinsic = datasetHelper.nusc.get_sample_data(sample['data'][CAM_FRONT],
                                                                                    selected_anntokens=
                                                                                        visible_anno_tokens)
            im = Image.open(data_path)
            axes[2].imshow(im)
            axes[2].set_title("All annos except occluded anns(" + str(len(boxes))+")")
            axes[2].axis('off')
            axes[2].set_aspect('equal')
            for box in boxes:
                c = np.array(datasetHelper.nusc.colormap[box.name]) / 255.0
                box.render(axes[2], view=camera_intrinsic, normalize=True, colors=(c, c, c))

            plt.savefig(out_path)
