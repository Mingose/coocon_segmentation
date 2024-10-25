echo "hellow experiment"
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/yolact/a_yolact_r50_1xb8-55e_coco.py  --work-dir ./work_dir_10_12/yolact
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dir_10_12/mask-rcnn
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco.py     --work-dir ./work_dir_10_12/solov2
# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/solov2/a_solov3_r50_fpn_1x_coco.py     --work-dir ./work_dir_10_12/solov3
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt.py --work-dir ./work_dir_10_13/solov2-ResNeXt
# CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeSt.py --work-dir ./work_dir_10_13/solov2-ResNeSt

python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M1.py  --work-dir ./work_dir_10_13/solov2-ResNeXt_range_M1
python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco.py      --work-dir ./work_dir_10_13/solov2 
python tools/train.py configs/solov2/a_solov3_r50_fpn_1x_coco.py      --work-dir ./work_dir_10_13/solov3
python tools/train.py configs/yolact/a_yolact_r50_1xb8-55e_coco.py   --work-dir ./work_dir_10_13/yolact
python tools/train.py configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py  --work-dir ./work_dir_10_13/mask-rcnn

python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt.py   --work-dir ./work_dir_10_13/solov2-ResNeXt
python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_64.py  --work-dir ./work_dir_10_13/solov2-ResNeXt_64grid

python 
tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M2.py  --work-dir ./work_dir_10_13/solov2-ResNeXt_range_M2


# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/yolact/a_yolact_r50_1xb8-55e_coco.py  4 --work-dir ./work_dir_10_12/yolact
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py 4 --work-dir ./work_dir_10_12/mask-rcnn
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco.py    4  --work-dir ./work_dir_10_12/solov2 
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov3_r50_fpn_1x_coco.py     4  --work-dir ./work_dir_10_12/solov3
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt.py  4 --work-dir ./work_dir_10_12/solov2-ResNeXt
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_64.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_64grid
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M1.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_range_M1
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt_range_M2.py 4 --work-dir ./work_dir_10_12/solov2-ResNeXt_range_M2
