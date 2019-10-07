
# Experiments
---
## My Scenario
1. Use Stratified Shuffle Split to split all the data into training and validation dataset, with 7 : 3 proportion (random seed is set to 1).
2. The original image shape is 450x600, I resize all the image into 224x224 (The minimum size to use resnet50 with pretrained weight)
3. Use Adam Optimizer.


## Resnet18
A light weighted model to quick verify preprocessing and modeling (including learning rate, batch size, optimizer, training loss etc.)
 - Version 1
     - 4 image augmentation: `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate` and  `RandomBrightnessContrast`
     - Train model with original dataset, without oversample or undersample.
     - Learning rate set to 0.001 with batch size 64. Observed the loss in validation data is not stable, should larger the batch size or smaller learning rate.
     - Weird prediction, all the predictions are `bkl` or `nv`. Guess this is caused by imbalanced data.
 - Version 2
     - **Removed `RandomBrightnessContrast` from image augmentation**, since most of the images have close brightness. And remove this augmentation also faster the convergance of cross entropy loss.
     - Since get the weird result from version 1, implement oversampling on this version.
     - Still got weird prediction.
 - Version 3
     - If training and testing data are exactly the same, but got different result from `model.train()` and `model.eval()` mode. 
     - Didn't have enough time to fix this problem, I'll look into this problem later on.

## SE-ResneXt50 (32x4d)

A heavier weighted model, but still light enough to train on my GTX 1080Ti. And SE-ResneXt50 (32x4d) got the lowest top-1 error on Imagenet among the light weighted model.

 - Version 1
     - Same settings with `Resnet18 Version 2`, make sure the weird prediction is not because of my code.
     - **Batch size changed to 32**, due to memory issue. We can see that the validation loss become very unstable.
 - Version 2
     - Since accuracy can't properly represent the performance of model, So start I implement **Average Precision** to compare different models.
     - We can see that the validation loss are very unstable, since we haven't solve the batch size issue, (or reduce the learning rate).
 - Version 3
     - From the Resnet18 models, we found that batch size smaller than 64 will cause the training loss unstable. Implement **accumulation gradient**.
     - So I updated the gradient for every 4 batch, this equivalent to set the batch size to 128. The precision in validation data become more robust.
     - Implement the **weighted cross entropy loss**, the weight was determined by the insight from confusion matrix.
 - Version 4
     - We could found that the validation loss after 5 epochs become unstable in previous version. I believe it's because the learning rate decay is not fast enough, I added **Cosine Annealing Learning Rate** to faster the learning rate decay in later gradient update.
     - Change the weighted cross entropy loss to **weighted focal loss**, want to put more emphasize on dealing imbalanced data problem. But didn't see significant improvement.
     - Start to use previous version's best weight as initial weight, this is close to the concept of **Cyclic Learning Rate (CLR)**.
 - Version 5
     - Use previous best weight as initial weight.
     - Changed Average Precision to **Average F1 score**. After a long time consideration, I can't tell precision or recall is better in this scenario, so I decided to use F1 score.
     - Add **Test Time Augmentation (TTA)**, I did the same augmentation in training data when predicting. Then blend the probability given from different augmented validation image. In this case I use original image + 3 augmented image.
     - Based on the confusion matrix in Version 4, modified the weight of loss function.


| SE-ResneXt50 (32x4d) | (Weighted) Loss | Accuracy | Avg Precision / F-score |
|----------------------|-----------------|----------|-------------------------|
| Ver 1                | 0.6455 (CE)     | 0.7667   |                         |
| Ver 2                | 1.1789 (CE)     | 0.6974   | 0.6445 ( P)              |
| Ver 3                | 0.8874 (WCE)    | 0.7533   | 0.6333 ( P)              |
| Ver 4                | 22.8558 (WF)    | 0.8607   | 0.5732 ( P)              |
| Ver 5                | 7.1692 (WF)     | 0.8670   | 0.5751 ( F)              |


## SE-Resnet50

Chose an architecture similar to SE-ResneXt50 (32x4d) to compare with the F1 score and confusion matrix.

 - Version 1
     - Initialize the weight on Imagenet pretrain model.
     - Run exactly the same configurations with SE-ResneXt50 (32x4d.)


|                            | Weighted Focal Loss | Accuracy | Avg F-score |
|----------------------------|---------------------|----------|-------------|
| SE-ResneXt50 (32x4d) Ver 5 | 7.1692              | 0.8670   | 0.5751      |
| SE-Resnet50 Ver 1          | 15.2528             | 0.8312   | 0.5197      |


## Conclusion
1. Focal loss didn't improve the precision much as expected, put weighted on loss can improve significantly.
2. TTA, accumulation gradient and pretrain really helps on this case.
3. In the future work, we can test other augmentations like `ShiftScaleRotate`, `RandomScale` or `Random add black or gray line` (simulate hair).
4. The performance of two model (SE-ResneXt50 and SE-Resnet50) in HAM10000 dataset is consist with Imagenet benchmark.