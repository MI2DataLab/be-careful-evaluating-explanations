# Be careful when evaluating explanations regarding ground truth

Trained models checkpoints are available at [Google Drive](https://drive.google.com/file/d/1KdyqNqSyBaN8ja579e4KcEH9fYdKNsWU/view?usp=share_link)

![](imgs/example.png)

## Test set of CheXlocalize classes counts
| Target | No Finding count | Enlarged Cardiomediastinum count | Cardiomegaly count | Lung Opacity count | Lung Lesion count | Edema count | Consolidation count | Pneumonia count | Atelectasis count | Pneumothorax count | Pleural Effusion count | Pleural Other count | Fracture count | Support Devices count |
| -----: | ---------------: | -------------------------------: | -----------------: | -----------------: | ----------------: | ----------: | ------------------: | --------------: | ----------------: | -----------------: | ---------------------: | ------------------: | -------------: | --------------------: |
|      0 |              559 |                              370 |                493 |                358 |               654 |         583 |                 633 |             654 |               490 |                658 |                    548 |                 660 |            662 |                   353 |
|      1 |              109 |                              298 |                175 |                310 |                14 |          85 |                  35 |              14 |               178 |                 10 |                    120 |                   8 |              6 |                   315 |

## Reproducing results
* Unpack models in main folder.
* Download CheXlocalize dataset and put it into `dataset` folder (path should look like this: `dataset/chexlocalize`).
* To reproduce our results prepare python environment and install packages from `requirements.txt` file. 
* After setting up environment modify 2nd. line of `calculate_model_results.sh` to activate your python environment. 
* This file will calculate scores and GradCam saliency maps from pretrained models.
* Create second python environment which will calculate scores based on CheXlocalize code (requirements are at `scripts/chexlocalize_scripts/requirements.txt`).
* Now modify 2nd line of `calculate_heatmaps_scores.sh` to activate second python environment. This script calculates all heatmaps scores.