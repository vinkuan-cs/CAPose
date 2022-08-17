from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os

class COCODataset():
    '''
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    },
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    # cfg.DATASET.ROOT = data/coco/, TRAIN_SET = train2017
    def __init__(self, root, image_set):
        super().__init__(root, image_set)
        self.coco = COCO(self._get_ann_file_keypoint())
        self.image_set_index = self._load_image_set_index()                     # 为了读图片的注释
    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'image_info'
        # print(prefix, self.image_set)
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '.json'
        )

if __name__ == '__main__':
    COCODataset(root='../../data/coco/small',image_set='trian_2017')