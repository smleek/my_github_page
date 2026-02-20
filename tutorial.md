---
title: "Tutorial Blog - February 20, 2026"
---
# Random Forest for Classification

## Introduction

Random forest is a popular machine learning method used for classification and regression. Its advantage over logistic regression is that it doesn't require fulfilled assumptions for normal distribution like "no outliers" and it does not assume linearity between the predictors and log-odds of the response, strict independence of errors, or absence of multicollinearity. 

We will be using it here to classify patients as having diabetes or not. I was given this dataset as part of my Introduction to Regression course taken Fall 2025, where we used logistic regression to classify patients as having diabetes or not. 

## not sure what to put here

### Imports, setup, cleaning...

Starting with imports: Since I'm most familiar with pandas, that's what I'll be using for this tutorial. We also need to import the scikit-learn package, which is a very popular Python package for data science. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

I'll provide the diabetes dataset... somehow. The dataset is already pretty tidy, but I decided to drop the "row" column as I found it redundant. There isn't any real cleaning to do. 

diabetes = pd.read_csv("/path/to/Diabetes.txt", sep = r"\s+")
diabetes = diabetes.drop(columns = ['row'])

At this point, diabetes should output this: 

![diabetes dataset printout](/home/sujin/stat386/my_github_page/images/cleaned_diabetes.png)

### More setup...

We're going to isolate our response variable and predictor variables to make life easier. (So far, this is just like logistic regression.)

predictors = diabetes.drop(columns=['diabetes'])
response = diabetes['diabetes']

Good practice in machine learning is to split up data before training and evaluating a model for later model validation. Scikit-learn makes this really easy. Here, I've chosen the variable names pred_train, pred_test, response_train, and response_test for the output of the train_test_split function. The function outputs ???. I've chosen a test size of 0.2 for an 80-20 split, which is pretty standard. random_state is the same thing as set.seed(); 

pred_train, pred_test, response_train, response_test = train_test_split(predictors, response, test_size=0.2, random_state=1)

### Fitting the model

Now it's time to fit our model! This is really easy thanks to the scikit-learn package.

random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
random_forest.fit(pred_train, response_train)

This should output the below image, which tells you something but I don't know...... 

### Making predictions

This will output an array of 1s and 0s, where 1 means YES! THE PATIENT HAS DIABETES! and 0 means NO, THE PATIENT DOES NOT HAVE DIABETES. Note that the predictions are done on the earlier sectioned-out pred_test subset of the data, where pred_test is 20% of the data. 

predictions = random_forest.predict(pred_test)
predictions

### Evaluation

If you set your random_state = 1, then the results should look identical to mine. If not, they're probably pretty similar. 

![model evaluation](/home/sujin/stat386/my_github_page/images/evaluation.png)

Accuracy is around 75%, which is certainly better than 50%, which is the baseline for deciding whether a classification model is good or not. Interestingly, the model I fit last semester using logistic regression was substantially better with an accuracy of 80.6%.

## Conclusion 

conclusion and call to action

So, it turns out I picked not the greatest dataset to show off random forest with, since logistic regression seems to have done better than random forest. However, we know that random forest is more robust to overfitting, and we can now try this new method on other datasets. It's also important to note that random forest can classify into more than two bins. Try out random forest on a dataset you're curious about! 

---

*This tutorial was created as an assignment for STAT 386.*
