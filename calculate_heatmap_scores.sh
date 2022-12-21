#!/bin/bash
source your/python/chexlocalize/environment
## eval hitrate
python scripts/chexlocalize_scripts/eval.py --metric hitmiss --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'gradcam_maps/negative_regularization/' --save_dir 'negative_regularization'
python scripts/chexlocalize_scripts/eval.py --metric hitmiss --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'gradcam_maps/positive_regularization/' --save_dir 'positive_regularization'
python scripts/chexlocalize_scripts/eval.py --metric hitmiss --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'gradcam_maps/no_regularization/' --save_dir 'no_regularization'

## heatmaps tune
python scripts/chexlocalize_scripts/tune_heatmap_threshold.py --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --map_dir 'gradcam_maps/negative_regularization/' --save_dir 'negative_regularization'
python scripts/chexlocalize_scripts/tune_heatmap_threshold.py --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --map_dir 'gradcam_maps/positive_regularization/' --save_dir 'positive_regularization'
python scripts/chexlocalize_scripts/tune_heatmap_threshold.py --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --map_dir 'gradcam_maps/no_regularization/' --save_dir 'no_regularization'

## heatmap_to_segmentation
python scripts/chexlocalize_scripts/heatmap_to_segmentation.py --map_dir 'gradcam_maps/negative_regularization/' --output_path "negative_regularization/saliency_segmentations.json" --threshold_path "negative_regularization/tuning_results.csv" 
python scripts/chexlocalize_scripts/heatmap_to_segmentation.py --map_dir 'gradcam_maps/positive_regularization/' --output_path "positive_regularization/saliency_segmentations.json" --threshold_path "positive_regularization/tuning_results.csv" 
python scripts/chexlocalize_scripts/heatmap_to_segmentation.py --map_dir 'gradcam_maps/no_regularization/' --output_path "no_regularization/saliency_segmentations.json" --threshold_path "no_regularization/tuning_results.csv" 

## eval saliency_segmentations
python scripts/chexlocalize_scripts/eval.py --metric iou --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'negative_regularization/saliency_segmentations.json' --save_dir 'negative_regularization'
python scripts/chexlocalize_scripts/eval.py --metric iou --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'positive_regularization/saliency_segmentations.json' --save_dir 'positive_regularization'
python scripts/chexlocalize_scripts/eval.py --metric iou --gt_path 'dataset/chexlocalize/CheXlocalize/gt_segmentations_val.json' --pred_path 'no_regularization/saliency_segmentations.json' --save_dir 'no_regularization'

python scripts/summarize_mask_scores.py
