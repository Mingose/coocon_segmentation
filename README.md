```
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
### 数据集（未上传）
默认使用：data/Cocoon_Morbid_coco

## 调用已经训练好的模型推理(未上传)
'''sh
python demo/video_demo.py demo/1.mp4 --out outputs/test.mp4 --config_bbox configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py --config_mask mask_configs/cocoon/mask-rcnn_r50_fpn_1x_cocoon.py --checkpoint_bbox work_dirs/faster-rcnn_Cocoon_all/epoch_200.pth --checkpoint_mask work_dirs/mask-rcnn_Cocoon_all/epoch_100.pth
'''

## 训练faster-rcnn：
'''sh
python tools/train.py configs/faster_rcnn/aaa_faster-rcnn_r50_fpn_1x_coco.py
'''



## 测试faster：
'''sh
python demo/video_demo.py demo/1.mp4 configs/faster_rcnn/aaa_faster-rcnn_r50_fpn_1x_coco.py work_dirs/aaa_faster-rcnn_r50_fpn_1x_coco/epoch_3.pth --out 1_VIS.mp4
'''


## 训练mask：
'''sh
python tools/train.py configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py

CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py 4

'''

## 测试mask：
'''
python demo/video_demo.py demo/1.mp4 configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py work_dirs/a_mask-rcnn_r50_fpn_1x_coco/epoch_12.pth --out demo/1_mask.mp4
'''

## 所有更新的配置文件位置参照configs/*，修改找到对应config即可
以下是使用4卡训练，如果只想用一张卡，可以参照前面的例子修改。
'''sh
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco.py    4  --work-dir ./work_dir_10_12/solov2 
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov3_r50_fpn_1x_coco.py     4  --work-dir ./work_dir_10_12/solov3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt.py  4 --work-dir ./work_dir_10_12/solov2-ResNeXt
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_64.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_64grid
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M1.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_range_M1
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M2.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_range_M2
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M3.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_range_M3
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M4.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_range_M4
'''

