# Setup
library(tidyverse)
library(gridExtra)
library(MASS)
library(pROC)
library(arm)
library(randomForest)
library(dplyr)
library(Metrics)
library(caret)
library(ROCR)
library(randomForest)
library(ggplot2)

library(car)
library(RCurl)

# dataset: The Wisconsin Breast Cancer dataset-http://archive.ics.uci.edu/ml
# goal: predict the observations (i.e. tumors) are malignant or benign

# read the dataset
URL <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')

names <- c('id_number', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 
           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 
           'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 
           'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
           'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 
           'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
           'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
           'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst')

breast_cancer <- read.table(textConnection(URL), sep = ',', col.names = names)

# Tidy the data
str(breast_cancer)
breast_cancer$id_number <- as.character(breast_cancer$id_number)
sum(is.na(breast_cancer))
summary(breast_cancer)
# changed the data type of id_number to be character
# There is not any missing value here. Class distribution: 357 benign, 212 malignant.

# Split the data into a training and validation set
breast_cancer <- mutate(breast_cancer, bi_diagnosis = ifelse(diagnosis=="B", 0, 1))
breast_cancer$bi_diagnosis <- as.factor(breast_cancer$bi_diagnosis)
set.seed(10) #for repeatability
n=0.3*nrow(breast_cancer) 
validation.index=sample(1:nrow(breast_cancer),n) 
train=breast_cancer[-validation.index,] 
validation=breast_cancer[validation.index,]

# Fit a logistic regression model
logit <- glm(bi_diagnosis ~ radius_mean + texture_mean + perimeter_mean + area_mean + 
               smoothness_mean + compactness_mean + concavity_mean + concave_points_mean + 
               symmetry_mean + fractal_dimension_mean, 
             train, family = binomial)
summary(logit)

validation$yhat<- predict(logit, newdata=validation, type = 'response')
validation$classification <- ifelse(validation$yhat>0.5, 1, 0)
#validation$bi_diagnosis <- validation$diagnosis == "B"
confusionMatrix(factor(validation$bi_diagnosis), factor(validation$classification))
# From the logistic regression model, we found that texture_mean, smoothness_mean and symmetry_mean are three important variables. 
# From the confision matrix, we know that the accuracy of this model is about 93.5%. TPR = 61/(6+61) = 91%, FPR = 5/(5+98) =4.9%. 4.9% of benign observations in the test set are classified to be malignant. About (1-91%) 9% of observations in the test set are malignant, but they are classified to be benign.


