# Lesion Diagnosis

Automated predictions of disease classification within dermoscopic images.

## Problem Statement

Build and evaluate a deep learning model that classifies a dermoscopic image as one of the following classes:

- [Melanoma](https://dermoscopedia.org/Melanoma)
- [Melanocytic nevus](https://dermoscopedia.org/Benign_Melanocytic_lesions)
- [Basal cell carcinoma](https://dermoscopedia.org/Basal_cell_carcinoma)
- [Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)](https://dermoscopedia.org/Actinic_keratosis_/_Bowen%27s_disease_/_keratoacanthoma_/_squamous_cell_carcinoma)
- [Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)](https://dermoscopedia.org/Solar_lentigines_/_seborrheic_keratoses_/_lichen_planus-like_keratosis)
- [Dermatofibroma](https://dermoscopedia.org/Dermatofibromas)
- [Vascular lesion](https://dermoscopedia.org/Vascular_lesions) 

## Data

As data, use [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

For your convenience we also provide the dataset for easy download via these three links:

* [HAM10000_images_part_1.zip](https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_1.zip)
* [HAM10000_images_part_2.zip](https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_2.zip)
* [HAM10000_metadata.csv](https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_metadata.csv)

Example images for the different lesion categories:
![Example images of the different lesion categories.](lesions.png)

## Assignment Details

Feel free to use the existing implementation in this repository as a base, but beware of low code quality and some data scientific problems in this implementation, that you will need to fix. It is perfectly fine to solve the task in a programming language and framework of your own choice instead.

Here are some ideas on what could be interesting things to consider, you won't have time to go deep into all of them, so choose the areas that you find most interesting to implement and investigate. 

* Define and implement metrics suitable for this problem.
* Try different model architectures / hyper parameter settings and compare their performance.
* There are much more examples of some of the classes in the data set. How does 
that impact the way you approach this problem?
* There are not that many examples to learn from. What alternatives are there to 
train a good model despite not that much data?
* Improve the code quality, e.g. by following 
  [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
* You're free to split the dataset however you choose but motivate your decisions. 
   
Don't forget to keep track of some of your mishaps as well as the successful 
experiments. The work test is not about building the perfect classifier - we 
are more interested in how you approach the problem. 

If you have access to a GPU you are free to use it; alternatively you can use [Google Colab's GPU runtime](https://colab.research.google.com/), which is currently free of charge.
Make sure to save locally all of the output you will need to prepare the report (in case of Colab, local storage is not persistent).

## Deliverables 

Git commit your results to this repo when you're done with the assignment.
Also create a report describing your work and results.
Make sure to include descriptions on what you have done, including your modeling
choices, results, conclusions and visualizations.
Notebooks can be a good way to show and visualize your work and results, but
you are free to use alternative solutions. 

Zip up the repo and send us the file in an e-mail to ml-team@peltarion.com.

If your solution is good you will be invited to Peltarion’s office to present your work for the ML team and have a follow-up discussion.

## Questions

If you run into problems or have questions, don't hesitate to email ml-team@peltarion.com. Asking questions is a good thing.

## Appendix - If you want to use Colab for model training
After navigating to [Colab](https://colab.research.google.com/), start a new Python 3 notebook. In the Runtime menu, select Change runtime type and choose GPU as Hardware accelerator.

You can then run the following list of commands to download the data.

```bash
!wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_1.zip
!wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_images_part_2.zip
!wget https://storage.googleapis.com/peltarion-ml-assignment/HAM10000/HAM10000_metadata.csv
!unzip -qq HAM10000_images_part_1.zip -d data
!unzip -qq HAM10000_images_part_2.zip -d data
!mv HAM10000_metadata.csv data/

```

Once you have uploaded the code in the zip file to your Colab notebook environment you can use

```bash
!python main.py
```

to run the training loop.
