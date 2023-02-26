import os, sys
import numpy as np
import json
import cv2 

# box color
color = [ 
(128,0,0),
(139,0,0),
(165,42,42),
(178,34,34),
(220,20,60),
(255,0,0),
(255,99,71),
(255,127,80),
(205,92,92),
(240,128,128),
(233,150,122),
(250,128,114),
(255,160,122),
(255,69,0),
(255,140,0),
(255,165,0),
(255,215,0),
(184,134,11),
(218,165,32),
(238,232,170),
(189,183,107),
(240,230,140),
(128,128,0),
(255,255,0),
(154,205,50),
(85,107,47),
(107,142,35),
(124,252,0),
(127,255,0),
(173,255,47),
(0,100,0),
(0,128,0),
(34,139,34),
(0,255,0),
(50,205,50),
(144,238,144),
(152,251,152),
(143,188,143),
(0,250,154),
(0,255,127),
(46,139,87),
(102,205,170),
(60,179,113),
(32,178,170),
(47,79,79),
(0,128,128),
(0,139,139),
(0,255,255),
(0,255,255),
(224,255,255),
(0,206,209),
(64,224,208),
(72,209,204),
(175,238,238),
(127,255,212),
(176,224,230),
(95,158,160),
(70,130,180),
(100,149,237),
(0,191,255),
(30,144,255),
(173,216,230),
(135,206,235),
(135,206,250),
(25,25,112),
(0,0,128),
(0,0,139),
(0,0,205),
(0,0,255),
(65,105,225),
(138,43,226),
(75,0,130),
(72,61,139),
(106,90,205),
(123,104,238),
(147,112,219),
(139,0,139),
(148,0,211),
(153,50,204),
(186,85,211),
(128,0,128),
(216,191,216),
(221,160,221),
(238,130,238),
(255,0,255),
(218,112,214),
(199,21,133),
(219,112,147),
(255,20,147),
(255,105,180),
(255,182,193),
(255,192,203),
(250,235,215),
(245,245,220),
(255,228,196),
(255,235,205),
(245,222,179),
(255,248,220),
(255,250,205),
(250,250,210),
(255,255,224),
(139,69,19),
(160,82,45),
(210,105,30),
(205,133,63),
(244,164,96),
(222,184,135),
(210,180,140),
(188,143,143),
(255,228,181),
(255,222,173),
(255,218,185),
(255,228,225),
(255,240,245),
(250,240,230),
(253,245,230),
(255,239,213),
(255,245,238),
(245,255,250),
(112,128,144),
(119,136,153),
(176,196,222),
(230,230,250),
(255,250,240),
(240,248,255),
(248,248,255),
(240,255,240),
(255,255,240),
(240,255,255),
(255,250,250)]

img_folder = "coco/demo_val2017"
json_file ="output.bbox.json"
ann_file = "coco/annotations/instances_demo_val2017.json"

if not os.path.exists('result'):
        os.makedirs('result')


# %%

per_img = []
flag = 0
# load the eval res
with open(json_file, 'r') as f:
    coco_d = json.load(f)

for image_info in coco_d:
    # get 1 image
    if not flag or flag == image_info["image_id"]:
        per_img.append(image_info)

    else:
        img_name = str(flag).zfill(12)  + '.jpg'
        img_full = os.path.join(img_folder, img_name)
        img = cv2.imread(img_full)
        # add bbox and text(label) to the image 
        for bbox in per_img:
            
            if bbox['score'] >= 0.5:
                x, y, w, h = bbox['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(img, (x, y), (x+w,y+h), color[bbox['category_id']],2)
                cv2.putText(img, str(bbox['category_id'])+" :  "+str(round(bbox['score'],3)), (x, y-8), cv2.FONT_HERSHEY_COMPLEX, 0.3, color[bbox['category_id']], 1, cv2.LINE_AA)
        
        # cv2.imshow(img_name, img)
        # cv2.waitKey(0)
        # store the image
        cv2.imwrite('result/'+img_name, img)
        cv2.destroyWindow(img_name) 
        
        per_img = []
        per_img.append(image_info)

    flag = image_info['image_id']    

#%%
# load the gts
with open(ann_file, 'r') as f:
    coco_ann = json.load(f)

img_dict = {}
# get image info and store in dict
for image_info in coco_ann['annotations']:
    
    img_dict.setdefault(image_info['image_id'],[])
    img_dict[image_info['image_id']].append([image_info['bbox'],image_info['category_id']])

# operate for each image
for image_id in img_dict:
    
    img_name = str(image_id).zfill(12)  + '.jpg'
    img_full = os.path.join(img_folder, img_name)
    img = cv2.imread(img_full)
    # add bbox and text(label) to the image 
    for bbox in img_dict[image_id]:

        x, y, w, h = bbox[0]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x+w,y+h), color[bbox[1]],2)
        cv2.putText(img, str(bbox[1]), (x, y-8), cv2.FONT_HERSHEY_COMPLEX, 0.5, color[bbox[1]], 1, cv2.LINE_AA)
        
    # cv2.imshow("gt"+img_name, img)
    # cv2.waitKey(0)
    # store the image
    cv2.imwrite('result/'+'gt_'+img_name, img)
    cv2.destroyWindow("gt"+img_name)

