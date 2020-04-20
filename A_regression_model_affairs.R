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

library(AER)
library(car)
library(RCurl)

# the Fair's Affairs dataset in AER package

# describe
data(Affairs, package="AER")
str(Affairs)
dim(Affairs)
sum(is.na(Affairs))
head(Affairs)
summary(Affairs)
table(Affairs$affairs, Affairs$gender)
table(Affairs$affairs, Affairs$yearsmarried)

# The males in the survey are more likely to have affairs than females. The mean of affairs is 1.456 times, but the median of affairs is 0. It shows that some observations had many affairs. When the years of marriage increase, people are more likely to have affairs. 
# There are some privacy and ethical concerns here. For instance, it is self-reported. This means a man may tell a lie because he does not want others know his personal issues. Also, if others know that somebody is an observation in the survey, they may know which observation he or she is.



# explore the characteristics of participants who engage in extramarital sexual intercourse using binary outcome
Affairs <- mutate(Affairs, bi_affairs = ifelse(affairs==0, 0, 1))

# Use an regression model to explore the relationship between having an affair and other personal characteristics.
fit_all <- glm(bi_affairs ~ gender + age + yearsmarried + children + religiousness + 
                 education + occupation + rating, data = Affairs, family = binomial)
summary(fit_all)

# try to obtain a "best" fit model
library(bestglm)
Affairs_new <- Affairs[, c("gender","age","yearsmarried","children","religiousness",
                           "education", "occupation","rating","bi_affairs")]
res.bestglm <-
  bestglm(Xy = Affairs_new, IC = "AIC", family=binomial)
res.bestglm$BestModels
summary(res.bestglm$BestModel)

# best fit model
best_fit_reduce <- glm(bi_affairs ~ age + yearsmarried + religiousness + rating, 
                       data = Affairs, family = binomial)
summary(best_fit_reduce)


# Create an artificial test dataset where martial rating varies from 1 to 5 and all other variables are set to their means. Use this test dataset and the predict function to obtain predicted probabilities of having an affair for case in the test data.
Rating <- c(1,2,3,4,5)
test <- data.frame(age = mean(Affairs$age), yearsmarried = mean(Affairs$yearsmarried), 
                   religiousness = mean(Affairs$religiousness), rating = Rating)
test$prob <- predict(best_fit_reduce, newdata=test, type = 'response')

# visualization
plot(x=1:nrow(test), y=test$prob, main="Predictions on the Test Dataset", 
     xlab="index", ylab="Probability of a prediction" )











