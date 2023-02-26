# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Evaluation for FasterRcnn"""
import cv2
import os
import argparse
import time
import numpy as np
from pycocotools.coco import COCO
import mindspore.common.dtype as mstype
from mindspore import context, save_checkpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed, Parameter

from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.util import coco_eval, bbox2result_1image, results2json

set_seed(1)

parser = argparse.ArgumentParser(description="FasterRcnn evaluation")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
parser.add_argument("--ann_file", type=str, default="val.json", help="Ann file, default is val.json.")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--device_target", type=str, default="CPU",
                    help="device where the code will be implemented, default is CPU")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
args_opt = parser.parse_args()

# context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)
# PYNATIVE_MODE is more friendly for debug, but typically slower
context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)

def fasterrcnn_eval(dataset_path, ckpt_path, ann_file):
    """FasterRcnn evaluation."""
    rank = 0
    device_num = 1

    print("Start create dataset!", flush=True)
    # create the dataset, reference: train.py
    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(dataset_path, batch_size=config.test_batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing,
                                        is_training=False)
    dataset_coco = COCO(ann_file)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!", flush=True)

    # create the network
    net = Faster_Rcnn_Resnet50(config=config)

    # load checkpoints
    load_path = ckpt_path
    param_dict = load_checkpoint(load_path)
    load_param_into_net(net, param_dict)

    # set as eval mode
    net = net.set_train(False)

    # eval loop
    eval_iter = 0
    outputs = []

    max_num = 128
    for data in dataset.create_dict_iterator(num_epochs=1):
        eval_iter += 1
        print("{}-iteration.".format(eval_iter))
        # load image and gt
        img_data, img_shape = data['image'], data['image_shape']
        gt_bboxes, gt_labels, gt_num = data['box'], data['label'], data['valid_num']
        # get inference res
        res = net(img_data, img_shape, gt_bboxes, gt_labels, gt_num)

        res_bboxes, res_labels, res_masks = res[0], res[1], res[2]
        # operate for each eval image
        for i in range(config.test_batch_size):
            res_mask = np.squeeze(res_masks.asnumpy()[i, :, :])

            res_bbox = np.squeeze(res_bboxes.asnumpy()[i, :, :])[res_mask, :]
            res_label = np.squeeze(res_labels.asnumpy()[i, :, :])[res_mask]

            if res_bbox.shape[0] > max_num:
                idx = np.argsort(-res_bbox[:, -1])[:max_num]
                res_bbox, res_label = res_bbox[idx], res_label[idx]
            # get results from 1 image
            output_1image = bbox2result_1image(res_bbox, res_label, config.num_classes)
            outputs.append(output_1image)
    # store results on the test set
    result_files = results2json(dataset_coco, outputs, "./output")
    # eval the results
    coco_eval(result_files, ["bbox"], dataset_coco, single_result=True)


if __name__ == '__main__':
    prefix = "FasterRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    print("CHECKING MINDRECORD FILES ...", flush=True)

    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                print("Create Mindrecord. It may take some time.", flush=True)
                data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir), flush=True)
            else:
                print("coco_root not exits.", flush=True)
        else:
            if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
                print("Create Mindrecord. It may take some time.", flush=True)
                data_to_mindrecord_byte_image("other", False, prefix, file_num=1)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir), flush=True)
            else:
                print("IMAGE_DIR or ANNO_PATH not exits.", flush=True)

    print("CHECKING MINDRECORD FILES DONE!", flush=True)
    print("Start Eval!", flush=True)
    fasterrcnn_eval(mindrecord_file, args_opt.checkpoint_path, args_opt.ann_file)

