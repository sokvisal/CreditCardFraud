# CreditCardFraud


The aim of this project is to train a machine learning model to detect fraudulent credit card transactions. Building a robust model is important to protect customers from fraudulent purchases. The dataset is publicly available, and can be accessed [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Note that most of the variables represent principal components, obtained from PCA of the original variables (which are hidden as they are sensitive information).

This project looks at using Decision Tree and Random Forest Classifier models to identify fraudulent transactions. However, these two models alone are not effective as the number of fraudulent and non-fraudulent transactions are not similar, with fraudulent transactions representing only a small fraction of the dataset. In fact, the two models are therefore biased toward predicting non-fraudulent transactions. Here, I also explore using a technique known as SMOTE to oversample the fraudulent class. This oversampling technique relies on making synthetic observations of fraud transactions based on what is available in the dataset.
