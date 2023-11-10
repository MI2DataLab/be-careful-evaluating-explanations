# Commands that we run to calculate _mIoU_ and _hit rate_ scores
## eval hitrate
* `python eval.py --metric hitmiss --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'gradcam_maps/negative_regularization/' --save_dir 'negative_regularization'`
* `python eval.py --metric hitmiss --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'gradcam_maps/positive_regularization/' --save_dir 'positive_regularization'`
* `python eval.py --metric hitmiss --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'gradcam_maps/no_regularization/' --save_dir 'no_regularization'`

## heatmaps tune
* `python tune_heatmap_threshold.py --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --map_dir 'gradcam_maps/negative_regularization/' --save_dir 'negative_regularization'`
* `python tune_heatmap_threshold.py --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --map_dir 'gradcam_maps/positive_regularization/' --save_dir 'positive_regularization'`
* `python tune_heatmap_threshold.py --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --map_dir 'gradcam_maps/no_regularization/' --save_dir 'no_regularization'`

## heatmap_to_segmentation
* `python heatmap_to_segmentation.py --map_dir 'gradcam_maps/negative_regularization/' --output_path "negative_regularization/saliency_segmentations.json" --threshold_path "negative_regularization/tuning_results.csv"`
* `python heatmap_to_segmentation.py --map_dir 'gradcam_maps/positive_regularization/' --output_path "positive_regularization/saliency_segmentations.json" --threshold_path "positive_regularization/tuning_results.csv"`
* `python heatmap_to_segmentation.py --map_dir 'gradcam_maps/no_regularization/' --output_path "no_regularization/saliency_segmentations.json" --threshold_path "no_regularization/tuning_results.csv"`

## eval saliency_segmentations
* `python eval.py --metric iou --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'negative_regularization/saliency_segmentations.json' --save_dir 'negative_regularization'`
* `python eval.py --metric iou --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'positive_regularization/saliency_segmentations.json' --save_dir 'positive_regularization'`
* `python eval.py --metric iou --gt_path 'chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'no_regularization/saliency_segmentations.json' --save_dir 'no_regularization'`
