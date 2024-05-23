from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

try:
    from .generic_dataset_rgbt_gnnloss import GenericDataset_rgbt
except:
    from generic_dataset_rgbt_gnnloss import GenericDataset_rgbt


class RGB_T(GenericDataset_rgbt):
    num_classes = 1
    num_joints = 17
    default_resolution = [640, 1088]
    max_objs = 300
    class_name = ['person']
    cat_ids = {1: 1}

    def __init__(self, opt, split):
        super(RGB_T, self).__init__()
        data_dir = opt.data_dir
        if split == 'test' or split == 'val':
            img_dir = os.path.join(
                data_dir, 'test')
        elif split == 'train':
            img_dir = os.path.join(
                data_dir, 'train')

        if split == 'val':
            ann_path_r = os.path.join(data_dir, 'annotations',
                                      '{}_half.json').format(split)

            ann_path_i = os.path.join(data_dir, 'annotations',
                                      '{}_half.json').format(split)
        else:

            ann_path_r = os.path.join(data_dir, 'annotations',
                                      '{}.json').format(split)
            ann_path_i = os.path.join(data_dir, 'annotations_06',
                                      '{}.json').format(split)

        print('==> initializing MOT20 {} data.'.format(split))

        self.images = None
        # load image list and coco
        super(RGB_T, self).__init__(opt, split, ann_path_r, ann_path_i, img_dir)

        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def _save_results(self, records, fpath):
        with open(fpath, 'w') as fid:
            for record in records:
                line = json.dumps(record) + '\n'
                fid.write(line)
        return fpath

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes:
            if type(all_bboxes[image_id]) != type({}):
                # newest format
                dtboxes = []
                for j in range(len(all_bboxes[image_id])):
                    item = all_bboxes[image_id][j]
                    bbox = item['bbox']
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "tag": item['class'],
                        "box": bbox_out,
                        "score": float("{:.2f}".format(item['score']))
                    }
                    dtboxes.append(detection)
            img_info = self.coco.loadImgs(ids=[image_id])[0]
            file_name = img_info['file_name']
            detections.append({'ID': file_name[:-4], 'dtboxes': dtboxes})
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        self._save_results(self.convert_eval_format(results),
                           '{}/results_rgbt-single.odgt'.format(save_dir))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        try:
            os.system('python tools/crowdhuman_eval/demo.py ' + \
                      '../data/crowdhuman/annotation_val.odgt ' + \
                      '{}/results_crowdhuman.odgt'.format(save_dir))
        except:
            print('Crowdhuman evaluation not setup!')
