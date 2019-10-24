# Personal project for Skin Cancer MNIST:HAM10000

## Environments
```
Ubuntu 16.04.6 LTS
CUDA Version 10.1
Python 3.6.5
```

## Quick Start
```bash
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirments.txt

python main.py --backbone se_resnext50_34x4d
```

## Exploratory Data Analysis (EDA)
 - To understand more about the HAM10000 dataset and the prediction made from model, check out the following jupyter notebook:
     - EDA.ipynb
     - post_EDA.ipynb


## Preprocessing
 - Train-Validation splitting
     - At first, I thought it might be more reasonable to split the data by `lesion_id`. But I believed the images won't share the personal information, so it won't be necessary to split on `lesion_id`. I believe splitting the data without consider `lesion_id` should not have data leakage problem.
     - Due to the imbalanced data, I chose **stratified train test splitting** along with fixed random seed as my spliting strategy.
 - Imbalanced data
     - **Oversample** on the minority classes, after the sampling. All the classes have at least `args.imbalanced_weight` percent of the number of data in majority class. In my case, given $N_{maj}$ is the number of data in training, the other classes will oversample to at least $0.15N_{maj}$ samples.
     - Using [WeightedRandomSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) from pytorch, but it spent too much time on sampling, so I removed it.
 - Data augmentation
     - For data augmentation, I used package: [albumentations](https://github.com/albu/albumentations). At the beginning I chose these 4 augmentation methods: `HorizontalFlip`, `VerticalFlip`, `ShiftScaleRotate` and  `RandomBrightnessContrast`.
     - The reason why I chose these 4 augmentaion methods is based on **EDA** (for more detail check EDA.ipynb). After validation with resnet18, I removed `RandomBrightnessContrast` after `resnet18 ver2`, because this will slow down the weight convergance.

## Performance Measurement
 - Training objective function
     - Weighted Cross Entropy Loss
     - **Weighted Focal Loss**
 - Evaluation Metrics
     - Accuracy
     - Average Precision
     - **Average F-measure**

1. Since the serious imbalanced data issue, I added weights on each loss function. In my case, each of the weight is [1.0, 0.9, 0.9, 1.0, 1.0, 0.5, 0.7] corresponding to the classes ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']. This weights were decided by the observation on confusion matrix.
2. Weighted Cross Entropy loss significantly improved the average precision in validation data, while changing cross entropy loss to focal loss doesn't improve a lot.
3. Due to the imbalanced problem, the accuracy can't really represent the performance of the model. At first, I chose precision as my metrics, but I can't tell precision or recall is more important on this case, so I use **Average F-measure**.


## Model Training
To train the deep learning model, I used [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) to load the **imagenet** weights as my initial weights. For more detail, please check the README.md in the `experiment` folder.

- Resnet18:
    - Chose a light weighted model to find better **learning rate** and better **batch size** quickly.
    - Found a bug in torchvision models, I got different result from `model.train()` and `model.eval()` even if the inputs are the same. Not sure what cased this bug, I will use **MINST** dataset to verify this problem. My preliminary guess is torch 1.2.0 haven't fully support CUDA 10.1
 - SE-ResneXt50 (34x4d):
     - Se-ResneXt50 (34x4d) is [the state of the art CNN architecture on Imagenet](https://arxiv.org/pdf/1810.00736.pdf) in small model size. Since my GPU (GTX 1080Ti) has only 11 GB memory, light weighted model can train with larget batch size.
     - Added **accumulation gradient** to delay the gradient update. The reason I did so was that in the experiment of resnet18, I found that to stable the loss function curve, batch size had to be at least 128. With the help of accumulation gradient, I could achieve the same result as  batch size 128 through accumulation gradient along with batch size 32.
     - Added **Test Time Augmentation**(TTA) to improve the performance on Validation data.

 - SE-Resnet50:
     - Train a similar architecture with SE-ResneXt50 (34x4d) to compare with it. (All configurations are equal to SE-ResneXt50(34x4d) except I trained SE-Resnet50 with 20 epoches)

|                        | Focal Loss | Accuracy | Avg Fsocre |
|------------------------|------------|----------|------------|
| SE-ResneXt50 (32x4d)   |  7.4301  | 0.8612 | 0.5737   |
| SE-Resnet50            | 15.2528  | 0.8312 | 0.5197  |


## Future Work
 - Train an **Supervised AutoEncoder** or use [Data Shaply](https://arxiv.org/abs/1904.02868) to detect mislabel (low quality) images.
 - Use **Cyclic learning rate** to ensemble (blending) models.
 - Check the bug from torchvision pretrained model. Using MNIST official example to test.
 - Train more other SoTA model (FishNet, EfficientNet, ...)

## Conclusion
1. For the imbalanced data, oversampling and weighted loss had significant improvement on validation metrics. On the other hand, focal loss only improved a little.
2. It's hard to tell whether some wrong predicted images were from mislabeling or weak performance of the model.
3. Two models' performance (SE-ResneXt50 (34x4d) and SE-Resnet50) on HAM10000 dataset is consistent with Imagenet dataset.


## References
1. [albumentations](https://github.com/albu/albumentations)
2. [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)
3. [Landmark2019-1st-and-3rd-Place-Solution](https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution)
4. [HAM10000 Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
5. [Benchmark Analysis of Representative
Deep Neural Network Architectures](https://arxiv.org/pdf/1810.00736.pdf)
