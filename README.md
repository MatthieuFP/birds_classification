## Introduction

This repository presents my work for the 2020 Kaggle Competition organized 
by the Object Recognition and Computer Vision class ([see here](https://www.kaggle.com/c/mva-recvis-2020/leaderboard)).

The goal of this project is to solve a fine-grained birds classification problem. More specifically, models have to 
obtain the best accuracy score on 20 classes of the [Caltech CUB200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

I built a Vision Transformer from [Pytorch-Image-model](https://github.com/rwightman/pytorch-image-models) stacked with Inceptionv3 to perform 0.8774 accuracy on 
the public leaderboard.
## Requirements

Python : 3.8.5
```
pip install -r requirements.txt
git clone https://github.com/Bjarten/early-stopping-pytorch early_stopping
```

## Preprocessing steps
Please refer to *bird_detector.ipynb* to crop the images around birds. It uses a pre-trained on COCO dataset Mask-RCNN implemented by Facebook [detectron2](https://github.com/facebookresearch/detectron2).
It creates a duplicate of *bird_dataset* named *cropped_birds* which has the same structure.

If it did not detect birds in all images :
```
python3 utils/correct_bounding_boxes.py 
```

## To train a model
### Supervized learning
```
python3 main.py --model stacked --data cropped_birds --epochs 50 --batch_size 4 --lr 2e-5 --patience 5 \
                --dropout .33 --weight_decay 1e-4 --cfg vit_large_patch16_224 --horizontal_flip 1 \
                --random_rotation 0 --erasing 0 --vertical_flip 1 --accumulation_steps 2 --optimizer adam
```
This file creates the folder **experiment** if not existing before and create automatically a folder *RUN_ID* (e.g. 42929)
located in the **experiment** folder and containing the *model.pt* saved, the *stdout.txt* and *results.json* files as well as
a *report* with the evolution of weights, gradients, losses and accuracy to visualize graphs on tensorboard.

### Semi-supervized learning 
(need a pre-trained model by main.py), inspired by [FixMatch](https://github.com/google-research/fixmatch)
```
python3 pseudo_labelling.py --RUN_ID 1e93b --model stacked --batch_size 4 --epochs 50 --patience 4 --lr 5e-6 \
                            --weight_decay 1e-4 --dropout 0.25 --cfg vit_large_patch16_224 --horizontal_flip 1 \
                            --vertical_flip 1 --random_rotation 0 --erasing 0 --size 224 --accumulation_steps 2 \
                            --threshold 0.9 --strong_augmentation 1 --T2 10 --factor 2 --prob 0.66 --log_interval 1000 \
                            --experiment experiment --smooth_prob 0.14 --loss_smoothing 1
```
This file creates the folder **semi_supervized_experiment** if not existing before and create automatically a folder ***RUN_ID*** (e.g. 42929)
located in the **semi_supervized_experiment** folder and containing the *model.pt* saved, the *stdout.txt* and *results.json* files as well as
a *report* with the evolution of weights, gradients, losses and accuracy to visualize graphs on tensorboard.



## Evaluation

```
python3 evaluate.py --data cropped_birds --model stacked --cfg vit_large_patch16_224 --experiment experiment \
                    --RUN_ID 42929 --size 224
```
- --experiment : folder containing the model to evaluate
- --RUN_ID : id of the model trained by main.py or pseudo_labelling.py files
- --model : model type

## Classification report

```
python3 miss_correct.py --data cropped_birds --model stacked --cfg vit_large_patch16_224 --experiment experiment \
                        --RUN_ID 42929 --size 224
```
Print a classification report with recall and precision of the validation images for all classes.

Display misclassified validation images, the corresponding probabilities, the predicted class and the ground truth class.

## References

<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.

<a id="2">[2]</a> 
Yuxin Wu and Alexander Kirillov and Francisco Massa and Wan-Yen Lo and Ross Girshick, 2019,
[https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2).

<a id="3">[3]</a> 
C. Ledig et al., 
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, 
CVPR,
2017,
105-114.
 
<a id="4">[4]</a> 
Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil,
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,
ICLR, 2021.

<a id="5">[5]</a> 
Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel,
FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence, 2020,
arXiv preprint arXiv:2001.07685.

