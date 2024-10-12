# python tools/train.py configs/rtmdet/a_rtmdet_tiny_8xb32-300e_coco.py --work-dir ./work_dir_10_11/rtmdet    
# python tools/train.py configs/cascade_rcnn/a_cascade-rcnn_r50_fpn_1x_coco.py    --work-dir ./work_dir_10_11/cascade
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov3_r50_fpn_1x_coco.py     --work-dir ./work_dir_10_11/solov3
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/yolact/a_yolact_r50_1xb8-55e_coco.py  --work-dir ./work_dir_10_11/yolact
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco.py     --work-dir ./work_dir_10_11/solov2
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dir_10_11/mask-rcnn
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/mask_rcnn/a_mask-rcnn_hrnet_1x_coco.py --work-dir ./work_dir_10_11/hrnet


CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeXt.py --work-dir ./work_dir_10_11/solov2-ResNeXt

CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco_ResNeSt.py --work-dir ./work_dir_10_11/solov2-ResNeSt
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/rtmdet/a_rtmdet_tiny_8xb32-300e_coco.py 4 --work-dir ./work_dir_10_10
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/cascade_rcnn/a_cascade-rcnn_r50_fpn_1x_coco.py 4 --work-dir ./work_dir_10_10
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/yolact/a_yolact_r50_1xb8-55e_coco.py 4 --work-dir ./work_dir_10_10
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/solov2/a_solov2_r50_fpn_1x_coco.py 4 --work-dir ./work_dir_10_10
# CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py 2 --work-dir ./work_dir_10_11/mask-rcnn

# 这个文件的配置是正确的，参考mask-rcnn对数据集的设置，帮我修改rtmdet的配置文件：



#2-mask-rcnn得到pkl，使用pkl推理画图，结果在results中
python tools/test.py \
    work_dir_10_11/mask-rcnn/a_mask-rcnn_r50_fpn_1x_coco.py \
    work_dir_10_11/mask-rcnn/epoch_20.pth    \
    --out work_dir_10_11/mask-rcnn/a_mask-rcnn_r50_fpn_1x_coco_test_results.pkl

python tools/analysis_tools/analyze_results.py \
    work_dir_10_11/mask-rcnn/a_mask-rcnn_r50_fpn_1x_coco.py \
    work_dir_10_11/mask-rcnn/a_mask-rcnn_r50_fpn_1x_coco_test_results.pkl \
    results/mask-rnn/ \
    --show-score-thr 0  \
    --show

python tools/analysis_tools/analyze_logs.py plot_curve work_dir_10_11/mask-rcnn/20241011_194409/20241011_194409.json --keys segm_mAP --legend segm_mAP --out comparison_bbox_mAP.png
##################################################

#3-mask-rcnn得到pkl，使用pkl推理画图，结果在results中
python tools/test.py \
    work_dir_10_11/solov2/a_solov2_r50_fpn_1x_coco.py \
    work_dir_10_11/solov2/epoch_1.pth    \
    --out work_dir_10_11/solov2_test_results.pkl

python tools/analysis_tools/analyze_results.py \
    work_dir_10_11/mask-rcnn/a_mask-rcnn_r50_fpn_1x_coco.py \
    work_dir_10_11/mask-rcnn/a_mask-rcnn_r50_fpn_1x_coco_test_results.pkl \
    results/mask-rnn/ \
    --show-score-thr 0  \
    --show


CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov3_r50_fpn_1x_coco.py      --work-dir ./work_dir_10_12/solov3
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/yolact/a_yolact_r50_1xb8-55e_coco.py --cfg-options max_epochs=100  --work-dir ./work_dir_10_12/yolact
CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/solov2/a_solov2_r50_fpn_1x_coco.py --cfg-options max_epochs=100     --work-dir ./work_dir_10_12/solov2
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/mask_rcnn/a_mask-rcnn_r50_fpn_1x_coco.py --work-dir ./work_dir_10_12/mask-rcnn