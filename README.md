# Offline Kannada handwritten character recognition using Manifold Smoothing and Label Propagation 


# About

Supervised learning techniques using deep learning models are highly effective in their application to handwritten character recognition. However, they require a large dataset of labelled samples to achieve good accuracies. 

For applications where particular classes of characters are difficult to train for, due to their rarity in occurrence, techniques from semi supervised learning and self supervised learning can be used to improve the accuracy in classifying these unusual classes. 

In this project, we analyze the effectiveness of using a combination of feature regularization, augmentation techniques and label propagation to classify previously unseen characters. 

We use a dataset of <b>Kannada Handwritten characters</b> to validate our approach. Kannada is the native language of the state of Karnataka, India. The Kannada script consists of <b>47 characters</b> (with exception of archaic characters za, la and ru) divided into 13 vowels ( called "Swaras") and <b>34 consonants</b> (called "Vyanjanas"). We focus on the identification of these characters from various sources where only a few characters are well represented and there's a need to incorporate new characters with limited labelled samples. 

We show the improvement in accuracy as well as reduced reliance on training data compared to previous approaches.

Read the PDF of the full report [here](Report.pdf).


# Proposed Method
We perform the following steps:
- Preprocessing and splitting of the dataset
- Pretraining the classification model
- Fine-tuning the classification model on novel dataset


# Preprocessing of the data
- The dataset consists of 47 classes, each having 400 samples (rescaled to 84px X 84px raster images)
- We split the dataset into 3 disjoint sets:
    - Base set: 24 classes
    - Novel set: 12 classes
    - Validation set: 11 classes
- Base set was used for pretraining (with all 400 samples)
- Novel set was used to check transfer learning capabilities
- Validation set consisted of 11 unseen classes used for hyperparameter search




