
## DAPS描述

DAPS[论文地址参见](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740295.pdf)是ECCV2022的中稿工作，论文全名为Domain Adaptive Person Search。该方法基于SeqNet，在CUHK-SYSU和PRW两个数据集之间测试了互相的迁移性能，并针对行人搜索这一任务提出了新的域自适应方法。

如下为MindSpore使用CUHK-SYSU数据集对DAPS进行训练的示例。该项目的Pytorch实现版本可以[参考](https://github.com/caposerenity/DAPS)

## 性能

|  Source   |  Target   | mAP  | Top-1 |                             
| :-------: | :-------: | :--: | :---: | 
|    PRW    | CUHK-SYSU | 78.5 | 80.7  | [ckpt](https://drive.google.com/file/d/1VFGiIqGI2SiJ98uIOnGLSqploWLX5AS_/view?usp=sharing) | [train_log](https://drive.google.com/file/d/1f-vGsN_wK08xUZF7R_thEfG18haOj-t6/view?usp=sharing) |
| CUHK-SYSU |    PRW    | 35.3 | 80.2  | [ckpt](https://drive.google.com/file/d/18eSJE3ljFl3SDf2H34PWVhFLmFhij3Rl/view?usp=sharing) | [train_log](https://drive.google.com/file/d/1DMPEqOu5pX2YLFRUFqQKNdAmA1YZDhqv/view?usp=sharing) |

## 数据集

使用的数据集：[CUHK-SYSU](https://github.com/ShuangLI59/person_search)和[PRW](https://github.com/liangzheng06/PRW-baseline)

# 环境要求

  - 硬件
    - 准备Ascend处理器搭建硬件环境。
  - 框架
    - [MindSpore](https://www.mindspore.cn/install/en)，本模型编写时版本为r1.2，12.30更新由r1.5编写的版本。
  - 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](

## 脚本及样例代码

```shell
.
└─project1_fasterrcnn
  ├─README.md                           
  ├─scripts
    └─run_eval_ascend.sh                
    └─run_eval_gpu.sh                   
    └─run_eval_cpu.sh                   
  ├─src
    ├─FasterRcnn
      ├─__init__.py                     
      ├─hm.py                           // reid所使用的混合memory定义
      ├─cluster.py                      // 聚类方法
      ├─jaccad.py                       // 计算jaccad距离
      ├─anchor_generator.py             // 生成anchor
      ├─faster_rcnn_r50.py              // 模型定义
      ├─fpn_neck.py                     // neck层
      ├─rcnn.py                         // head（检测和重识别头）
      ├─resnet50.py                     // ResNet-50
      ├─roi_align.py                    // ROI Align层
      └─rpn.py                          // Region Proposal Network
    ├─config.py               
    ├─dataset.py              
    ├─lr_schedule.py          
    ├─network_define.py       
    └─util.py                 
  ├─cocoapi                   
  ├─pretrained_faster_rcnn.ckpt         
  ├─eval.py                   // evaluation script
  └─train.py                  // training script
```

## 环境准备
```shell
pip install -r requirements.txt

# install COCO evaluation API
cd cocoapi/PythonAPI
python setup.py install
```

## 模型性能评测
执行如下命令
```shell
# evaluate (on Ascend/GPU/CPU. Choose one according to your device.)
sh ./scripts/run_eval_ascend.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
sh ./scripts/run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
sh ./scripts/run_eval_cpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH]
```

