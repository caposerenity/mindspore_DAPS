import time
import collections
from collections import OrderedDict
import numpy as np
import random
import mindspore

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def extract_dy_features(cfg, model, data_loader, device, is_source, memory_proposal_boxes=None, memory_target_features=None, momentum=0.2):
    model.eval()
    sour_fea_dict = collections.defaultdict(list)
    tgt_fea = OrderedDict()
    negative_fea = OrderedDict()
    positive_fea = OrderedDict()
    img_proposal_boxes = collections.defaultdict(list)
    img_store_idx = unqualified_nums  = 0
    for i, (images, targets) in enumerate(data_loader):
            
            #batch size是1
            gt_labels = targets[0]["labels"]
            gt_boxes = [t["boxes"] for t in targets]
            img_name = targets[0]["img_name"]

            if is_source:
                images, targets = model.transform(images, targets)
                features = model.backbone(images.tensors)
                box_features = model.roi_heads.box_roi_pool(features, gt_boxes, images.image_sizes)
                box_features = model.roi_heads.reid_head(box_features, is_source)
                embeddings, _ = model.roi_heads.embedding_head(box_features)
                embeddings = embeddings.data.cpu()
                for j in range(len(gt_boxes[0])):
                    sour_fea_dict[gt_labels[j].item()].append(embeddings[j].unsqueeze(0))

            else:
                
                detections = model(images,is_source=is_source)
                
                boxes = detections[0]["boxes"].data.cpu()
                embeddings = detections[0]["embeddings"].data.cpu()
                scores = detections[0]["scores"]
                if len(boxes)==0:
                    #print("Here is an image without qualified proposal")
                    orig_thresh = model.roi_heads.score_thresh
                    model.roi_heads.score_thresh = 0
                    detections = model(images,is_source=is_source)
                    boxes = detections[0]["boxes"].data.cpu()
                    embeddings = detections[0]["embeddings"].data.cpu()
                    img_proposal_boxes[img_name].append(boxes[0].numpy().tolist())
                    img_proposal_boxes[img_name] = mindspore.Tensor(img_proposal_boxes[img_name]).data.cpu()
                    tgt_fea[img_name+"_"+str(0)] = embeddings[0].unsqueeze(0)
                    model.roi_heads.score_thresh = orig_thresh
                    unqualified_nums+=1
                    continue

                inds = (scores>=cfg.EPS_P)
                hard_inds = (scores>=0.8) ^ inds
                all_embeddings = embeddings.clone()
                hard_embeddings = embeddings[hard_inds]
                embeddings = embeddings[inds]
                all_boxes = boxes.clone()
                hard_boxes = boxes[hard_inds]
                boxes = boxes[inds]
                ious = []
                for j in range(len(embeddings)):
                    if memory_proposal_boxes is None:
                        tgt_fea[img_name+"_"+str(j)] = embeddings[j].unsqueeze(0)
                        img_proposal_boxes[img_name].append(boxes[j].numpy().tolist())
                    else:
                        #iou mapping
                        ious.append([])
                        for k in range(len(memory_proposal_boxes[img_name])):
                            iou = _compute_iou(memory_proposal_boxes[img_name][k],boxes[j])
                            ious[-1].append(iou)
                        if max(ious[-1])>0.7:
                            ious[-1] = ious[-1].index(max(ious[-1]))
                            memory_target_features[img_store_idx+ious[-1]] = momentum * memory_target_features[img_store_idx+ious[-1]] + (1. - momentum) * embeddings[j]
                            try:
                                memory_proposal_boxes[img_name][ious[-1]] =  momentum * memory_proposal_boxes[img_name][ious[-1]] + (1. - momentum) * boxes[j]
                            except TypeError as e:
                                print(e)
                                print(boxes[j])
                                print(memory_proposal_boxes[img_name])
                                print(ious)
                        else:
                            ious[-1] = -1
                
                # delete unmapped bboxes in memory
                if memory_proposal_boxes is not None:
                    instance_id = 0
                    for idx in range(len(memory_proposal_boxes[img_name])):
                        if idx in ious:
                            img_proposal_boxes[img_name].append(memory_proposal_boxes[img_name][idx].numpy().tolist())
                            tgt_fea[img_name+"_"+str(instance_id)] = memory_target_features[img_store_idx+idx].unsqueeze(0)
                            instance_id+=1
                    # add unmapped high-confidence proposal
                    for idx in range(len(ious)):
                        if ious[idx]==-1:
                            img_proposal_boxes[img_name].append(boxes[idx].numpy().tolist())
                            tgt_fea[img_name+"_"+str(instance_id)] = embeddings[idx].unsqueeze(0)
                            instance_id+=1
                    img_store_idx+=len(memory_proposal_boxes[img_name])

                if len(img_proposal_boxes[img_name])==0:
                        img_proposal_boxes[img_name].append(all_boxes[0].numpy().tolist())
                        tgt_fea[img_name+"_"+str(0)] = all_embeddings[0].unsqueeze(0)
                        unqualified_nums += 1
                
                img_proposal_boxes[img_name] = mindspore.Tensor(img_proposal_boxes[img_name]).data.cpu()

                hard_negative = mindspore.Tensor.ones(hard_embeddings.shape[0])
                for j in range(hard_embeddings.shape[0]):
                    for k in range(len(img_proposal_boxes[img_name])):
                        if _compute_iou(hard_boxes[j],img_proposal_boxes[img_name][k])>cfg.HM_THRESH:
                            hard_negative[j] = 0
                            break
                hard_negative = (hard_negative>0)
                if hard_negative.sum()>0:
                    hard_embeddings = hard_embeddings[hard_negative]
                    negative_fea[img_name] = hard_embeddings

    if is_source:
        return sour_fea_dict
    else:
        print("unqualified_nums")
        print(unqualified_nums)
        return tgt_fea, img_proposal_boxes, negative_fea, positive_fea