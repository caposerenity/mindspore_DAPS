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

"""train FasterRcnn and get checkpoint files."""

import os
import argparse
import datetime
import os.path as osp
import time
import argparse
import ast
import numpy as np
import collections
from sklearn.cluster import DBSCAN
import mindspore
import mindspore.common.dtype as mstype
from mindspore import context, Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn import SGD
from mindspore.common import set_seed
from jaccad import compute_jaccard_distance
from src.FasterRcnn.faster_rcnn_r50 import Faster_Rcnn_Resnet50
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.config import config
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr
from src.cluster import extract_dy_features
from src.hm import HybridMemory
set_seed(1)

parser = argparse.ArgumentParser(description="FasterRcnn training")
parser.add_argument("--run_distribute", type=ast.literal_eval, default=False, help="Run distribute, default: false.")
parser.add_argument("--dataset", type=str, default="coco", help="Dataset name, default: coco.")
parser.add_argument("--pre_trained", type=str, default="", help="Pretrained file path.")
parser.add_argument("--device_target", type=str, default="Ascend",
                    help="device where the code will be implemented, default is Ascend")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")
parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default: 1.")
parser.add_argument("--rank_id", type=int, default=0, help="Rank id, default: 0.")
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=args_opt.device_id)


if __name__ == '__main__':
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            rank = args_opt.rank_id
            device_num = args_opt.device_num
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        else:
            init("nccl")
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in args_opt.mindrecord_dir,
    # and the file name is FasterRcnn.mindrecord0, 1, ... file_num.
    prefix = "FasterRcnn.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    print("CHECKING MINDRECORD FILES ...")

    if rank == 0 and not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if args_opt.dataset == "coco":
            if os.path.isdir(config.coco_root):
                if not os.path.exists(config.coco_root):
                    print("Please make sure config:coco_root is valid.")
                    raise ValueError(config.coco_root)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image("coco", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("coco_root not exits.")
        else:
            if os.path.isdir(config.image_dir) and os.path.exists(config.anno_path):
                if not os.path.exists(config.image_dir):
                    print("Please make sure config:image_dir is valid.")
                    raise ValueError(config.image_dir)
                print("Create Mindrecord. It may take some time.")
                data_to_mindrecord_byte_image("other", True, prefix)
                print("Create Mindrecord Done, at {}".format(mindrecord_dir))
            else:
                print("image_dir or anno_path not exits.")

    while not os.path.exists(mindrecord_file + ".db"):
        time.sleep(5)

    print("CHECKING MINDRECORD FILES DONE!")

    loss_scale = float(config.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    dataset = create_fasterrcnn_dataset(mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    target_dataset = create_fasterrcnn_dataset(mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank,
                                        num_parallel_workers=config.num_parallel_workers,
                                        python_multiprocessing=config.python_multiprocessing)

    source_classes = dataset_size = dataset.get_dataset_size()
    target_dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    net = Faster_Rcnn_Resnet50(config=config)
    net = net.set_train()

    load_path = args_opt.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)

        key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                       'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                       'down_sample_layer.0.weight': 'conv_down_sample.weight',
                       'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                       'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                       }
        for oldkey in list(param_dict.keys()):
            if not oldkey.startswith(('backbone', 'end_point', 'global_step', 'learning_rate', 'moments', 'momentum')):
                data = param_dict.pop(oldkey)
                newkey = 'backbone.' + oldkey
                param_dict[newkey] = data
                oldkey = newkey
            for k, v in key_mapping.items():
                if k in oldkey:
                    newkey = oldkey.replace(k, v)
                    param_dict[newkey] = param_dict.pop(oldkey)
                    break

        for item in list(param_dict.keys()):
            if not item.startswith('backbone'):
                param_dict.pop(item)

        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
        load_param_into_net(net, param_dict)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    memory = HybridMemory(256, source_classes, source_classes,
                            temp=0.05, momentum=0.2)
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)

    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if args_opt.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)
    start_epoch = 0
    for epoch in range(start_epoch, config.epoch_size):
        
        if (epoch==config.target_start_epoch):
            
            # DBSCAN cluster
            eps = 0.5
            eps_tight = eps-0.02
            eps_loose = eps+0.02
            print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1)
            cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1)

        if (epoch>=config.target_start_epoch):
            # init target domain instance level features
            # we can't use target domain GT detection box feature to init, this is only for measuring the upper bound of cluster performance
            #for dynamic clustering method, we use the proposal after several epoches for first init, moreover, we'll update the memory with proposal before each epoch
            print("==> Initialize target-domain instance features in the hybrid memory")
            tgt_cluster_loader = target_dataset
            if epoch==config.target_start_epoch:
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(config, model, tgt_cluster_loader, is_source=False)
            else:
                target_features = memory.features[source_classes:].data.cpu().clone()
                #target_features = memory.features[source_classes:source_classes+len(sorted_keys)].data.cpu().clone()
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(config, model, tgt_cluster_loader, is_source=False, memory_proposal_boxes=img_proposal_boxes, memory_target_features=target_features)
            sorted_keys = sorted(target_features.keys())
            print("target_features instances :"+str(len(sorted_keys)))
            target_features = mindspore.ops.cat([target_features[name] for name in sorted_keys], 0)
            target_features = mindspore.ops.normalize(target_features, dim=1).cuda()
            
            negative_fea = mindspore.ops.cat([negative_fea[name] for name in sorted(negative_fea.keys())], 0)
            print(negative_fea.shape)
            negative_fea = mindspore.ops.normalize(negative_fea, dim=1).cuda()
            print("hard negative instances :"+str(len(negative_fea)))

            source_centers = memory.features[0:source_classes].clone()
            memory.features = mindspore.ops.cat((source_centers, target_features), dim=0).cuda()
            del source_centers,target_features, tgt_cluster_loader
            
            # Calculate distance
            print('==> Create pseudo labels for unlabeled target domain with self-paced policy')
            target_features = memory.features[source_classes:].clone()

            rerank_dist = compute_jaccard_distance(target_features, k1=30, k2=6, search_option=3, use_float16=True)
            del target_features
            # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
            pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            print("pseudo_labels length :")
            print(len(pseudo_labels))
            print(pseudo_labels)
            num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
            num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)
            
            # generate new dataset and calculate cluster centers
            def generate_pseudo_labels(cluster_id, num):
                labels = []
                outliers = 0
                for i, id in enumerate(cluster_id):
                    if id!=-1:
                        labels.append(source_classes+id)
                    else:
                        labels.append(source_classes+num+outliers)
                        outliers += 1
                return mindspore.Tensor(labels).long()

            pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
            pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
            pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

            # compute R_indep and R_comp
            N = pseudo_labels.size(0)
            label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
            label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
            label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

            R_comp = 1-mindspore.ops.min(label_sim, label_sim_tight).sum(-1)/mindspore.ops.max(label_sim, label_sim_tight).sum(-1)
            R_indep = 1-mindspore.ops.min(label_sim, label_sim_loose).sum(-1)/mindspore.ops.max(label_sim, label_sim_loose).sum(-1)
            assert((R_comp.min()>=0) and (R_comp.max()<=1))
            assert((R_indep.min()>=0) and (R_indep.max()<=1))

            cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
            cluster_img_num = collections.defaultdict(int)
            for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
                cluster_R_comp[label.item()-source_classes].append(comp.item())
                cluster_R_indep[label.item()-source_classes].append(indep.item())
                cluster_img_num[label.item()-source_classes]+=1

            cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
            cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
            cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
            if (epoch==config.target_start_epoch):
                indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

            outliers = 0
            # use sorted_keys for searching pseudo_labels
            print('==> Modifying labels in target domain to build new training set')
            index_count = 0
            for i, anno in enumerate(target_dataset.annotations):
                boxes_nums = len(img_proposal_boxes[anno["img_name"]])
                anno["pids"]=mindspore.Tensor.zeros(boxes_nums)
                anno["boxes"]=img_proposal_boxes[anno["img_name"]]
                for j in range(boxes_nums):
                    index = sorted_keys.index(anno["img_name"]+"_"+str(j))
                    label = pseudo_labels[index]
                    indep_score = cluster_R_indep[label.item()-source_classes]
                    comp_score = R_comp[index]
                    if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()-source_classes])):
                        anno["pids"][j] = index_count+source_classes+1
                    else:
                        anno["pids"][j] = index_count+source_classes+1
                        pseudo_labels[index] = source_classes+len(cluster_R_indep)+outliers
                        outliers+=1
                    index_count += 1
                target_dataset.annotations[i] = anno
            print(index_count)
            # statistics of clusters and un-clustered instances
            '''index2label = collections.defaultdict(int)
            for label in pseudo_labels:
                index2label[label.item()]+=1
            print(sorted(index2label.items(), key=lambda d: d[1], reverse=True))
            index2label = np.fromiter(index2label.values(), dtype=float)
            print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                        .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))'''

            memory.features = mindspore.ops.cat((memory.features, negative_fea), dim=0).cuda()
            # hard_negative cases are assigned with unused labels
            memory.labels = (mindspore.ops.cat((mindspore.ops.arange(source_classes), pseudo_labels , mindspore.ops.arange(len(negative_fea))+pseudo_labels.max()+1)))
            memory.num_samples = memory.features.shape[0]
            print(len(memory.labels))
        else:
            memory.labels = (mindspore.ops.arange(source_classes))
            memory.num_samples = source_classes
    model.train(config.epoch_size, dataset, callbacks=cb)
