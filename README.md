```mermaid
graph TD;
    A[Input Image] --> B[Backbone (ResNet/ResNeSt/ResNeXt)]
    B --> C[Neck (FPN)]
    C --> D[Category Branch]
    C --> E[Mask Branch]
    E --> F[Kernel Branch]
    E --> G[Feature Branch]
    F --> H[Combine Kernel & Feature]
    H --> I[Final Mask Predictions]


```



## 调用已经训练好的模型推理
python demo/video_demo.py demo/1.mp4 --out outputs/test.mp4 --config_bbox configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --config_mask mask_configs/cocoon/mask-rcnn_r50_fpn_1x_cocoon.py --checkpoint_bbox work_dirs/faster-rcnn_Cocoon_all/epoch_200.pth --checkpoint_mask work_dirs/mask-rcnn_Cocoon_all/epoch_100.pth

## 训练faster-rcnn的是：
python tools/train.py configs/faster_rcnn/aaa_faster-rcnn_r50_fpn_1x_coco.py


## 测试faster是：
python demo/video_demo.py demo/1.mp4 configs/faster_rcnn/aaa_faster-rcnn_r50_fpn_1x_coco.py work_dirs/aaa_faster-rcnn_r50_fpn_1x_coco/epoch_3.pth --out 1_VIS.mp4

## 训练mask是：
python tools/train.py configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py 4

## 测试mask是：
python demo/video_demo.py demo/1.mp4 configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py work_dirs/a_mask-rcnn_r50_fpn_1x_coco/epoch_12.pth --out demo/1_mask.mp4


##


python tools/train.py configs/rtmdet/a_rtmdet_tiny_8xb32-300e_coco.py 
python tools/train.py configs/cascade_rcnn/a_cascade-rcnn_r50_fpn_1x_coco.py
python tools/train.py configs/yolact/a_yolact_r50_1xb8-55e_coco.py 

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py 4