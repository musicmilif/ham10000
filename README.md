# Peltarion assignment

## Environments
```
Ubuntu 16.04.6 LTS
CUDA Version 10.1
Python 3.6.5
```

## Quick Start
```bash
git clone git@github.com:musicmilif/peltarion.git
cd peltarion
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirments.txt

python main.py --backbone se_resnext50_34x4d
```

## Exploratory Data Analysis (EDA)
 - To understand more about the HAM10000 dataset and the prediction made from model check out the following jupyter notebook:
     - EDA.ipynb
     - post_EDA.ipynb


## Preprocessing
 - Train-Valid spliting
     - At first, I thought it might be more reasonable to split the data by `lesion_id`. But if the input of the model didn't have personal information, it won't be necessary to split on `lesion_id`. I believe there won't be data leakage.
     - Due to the imbalanced data, I chose **stratified train test spliting** with fixed random seed. 
 - Imbalanced data
     - **Oversample** on the minority classes, up to `args.imbalanced_weight` percent of the number of data in majority class. In my case, given $N_{nv}$ is the number of data in training, the other classes will oversample to at least $0.15N_{nv}$ samples.
     - Use [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) from pytorch, but it spent too much time on sampling, so I removed it.
 - Data augmentation
     - For data augmentation, I use package: [albumentations](https://github.com/albu/albumentations) to perform. At the beginning I chose 4 augmentations: `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate` and  `RandomBrightnessContrast`.
     - The reason why chose this 4 augmentaion is based on **EDA** (for more detail check EDA.ipynb), and after verifying by training the resnet18. I removed `RandomBrightnessContrast` after `resnet18 ver2`, because this will slow down the weight convergance.

## Performance Measurement
 - Training objective function
     - Weighted Cross Entropy Loss
     - **Weighted Focal Loss**
 - Evaluation Metrics
     - Accuracy
     - Average Precision
     - **Average F-measure**

1. Since the imbalanced data issue, I add weights on each loss function. In my case was [1.0, 0.9, 0.9, 1.0, 1.0, 0.5, 0.7] corresponding to classes ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], this weight was determined by the observation on confusion matrix.
2. Weighted Cross Entropy loss significantly improved the average precision in validation data. After changing cross entropy loss to focal loss doesn't improve a lot.
3. Due to imbalanced, accuracy can't really shows the performance of the model. At first, I chose precision as my metrics, but I can't tell precision or recall is more importance on this case, so I use **Average F-measure**.


## Model Training
To train the deep learning model, I use [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) to load the **imagenet** weights as my initial weight. For more detail, please check the README.md in the `experiment` folder.

- Resnet18:
    - Chose a light weighted model to quick verify **learning rate** and **batch size**.
    - Found a bug in torchvision models, I got different result from `model.train()` and `model.eval()` even if the inputs are the same. Not sure what cased this bug, I will use **MINST** dataset to verify this problem. My preliminary guess is torch 1.2.0 haven't fully support CUDA 10.1
 - SE-ResneXt50 (34x4d):
     - Se-ResneXt50 (34x4d) is [the state of the art CNN architecture on Imagenet](https://arxiv.org/pdf/1810.00736.pdf) in small model size. This model is perfect match with my GPU (since GTX 1080Ti only got 11 GB memory).
     - By the experience of resnet18, 128 batch size will have stable training loss curve. But my graphic card memory can only handel 32 batch size. So I added **accumulation gradient** to delay the gradient update.
     - Add **Test Time Augmentation**(TTA) to improve the performance on Validation data

 - SE-Resnet50:
     - Train a similar architecture with SE-ResneXt50 (34x4d) to compare with it. (All argument are equal to SE-ResneXt50(34x4d) except train SE-Resnet50 with 20 epoches)

|                        | Focal Loss | Accuracy | Avg Fsocre |
|------------------------|------------|----------|------------|
| SE-ResneXt50 (32x4d)   |  7.4301  | 0.8612 | 0.5737   |
| SE-Resnet50            | 15.2528  | 0.8312 | 0.5197  |


## Future Work
 - Train an **Supervised AutoEncoder** or use [Data Shaply](https://arxiv.org/abs/1904.02868) to detect mislabel (low quality) images.
 - Use **Cyclic learning rate** to ensemble (blending) models.
 - Check the bug from torchvision pretrained model. Using MNIST official example to test.
 - Train with other SoTA model (FishNet, EfficientNet, ...)

## Conclusion
1. To handle with imbalanced data, oversampling and weighted loss can solve the problem properly. But the performance of Focal loss was not significant.
2. Some wrong predicted images are hard to tell it's mislabel or weak performance of the model. It's necessary to use Data Shaply to check the quality of data.
3. Two models' performance (SE-ResneXt50 (34x4d) and SE-Resnet50) on HAM10000 dataset is consistant with Imagenet dataset.


## References
1. [albumentations](https://github.com/albu/albumentations)
2. [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)
3. [Landmark2019-1st-and-3rd-Place-Solution](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution)
4. [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
5. [Benchmark Analysis of Representative
Deep Neural Network Architectures](https://arxiv.org/pdf/1810.00736.pdf)