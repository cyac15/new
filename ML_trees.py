#!/usr/bin/env python
# coding: utf-8

# ##  Loading libraries: {-}

# In[26]:


import pandas as pd
import numpy as np
import os, glob



# ##  Part 1: Creating helper functions {-}
# 1. **Create a helper function to calculate Gini impurity called “calcGini”.**

# In[3]:


def calcGini(countClass1, countClass2):
    if (countClass1==0) | (countClass2 ==0):
        result = 0
    else:
        # do calcs
        prob1 = countClass1/(countClass1+countClass2)
        prob2 = countClass2/(countClass1+countClass2)        
        result = 1 - (np.square(prob1) + np.square(prob2))
    return result


# 2. **Create a helper function to calculate Gini impurity called “calcEntropy”.**

# In[4]:


def calcEntropy(countClass1, countClass2):   
    if (countClass1==0) | (countClass2 ==0):
        result = 0
    else:
        # do calcs
        prob1 = countClass1/(countClass1+countClass2)
        prob2 = countClass2/(countClass1+countClass2)        
        result = -(prob1*np.log2(prob1)+prob2*np.log2(prob2))
    return result


# 3. **Create a helper function to calculate weighted sums called “weightedSum”.**

# In[5]:


def weightedSum(listValue, listWeight):
    sum = 0
    for i in range(len(listValue)):
        tempResult = listValue[i]* listWeight[i]
        sum = sum + tempResult  
    return sum


# 4. **Verify that your functions are working as expected.**

# In[30]:


# Read "toyData.csv"
toyData = pd.read_csv("./toyData.csv")

# result = toyData[["Output"]].groupby(toyData["Variable A"]).count()

cols = toyData.columns[:3]

colName =[]
sumGini = []
sumEntropy = []

for col in cols:
    current = toyData[col]
    uniqueVals = current.unique()
    
    gini = []
    entropy = []
    weights = []
    
    for val in uniqueVals:
        # print(val)
        currentDf = toyData[toyData[col] == val]
        reds = len(currentDf[currentDf['Output'] == "Red"])
        blues = len(currentDf[currentDf['Output'] == "Blue"])
        # print(val, calcGini(reds, blues))
        gini.append(calcGini(reds, blues))
        
        # print(val, calcEntropy(reds, blues))
        entropy.append(calcEntropy(reds, blues))
        
        weights.append(len(currentDf)/len(current))
    print("column:", col)
    print("weights", weights)
    print("gini", gini)
    print("entropy", entropy, "\n")
    
    # calcweighted sum of gini
    #print(weightedSum(gini, weights))
    sumGini.append(weightedSum(gini, weights))
    
    #calcweighted sum of entropy
    #print("weightedSum of entropy: ", weightedSum(entropy, weights))
    sumEntropy.append(weightedSum(entropy, weights))
    colName.append(col)
roundSumGini = [round(num, 2) for num in sumGini]
roundSumEntropy = [round(num, 2) for num in sumEntropy]


output = {'Variable': colName,
          'Gini impurity': roundSumGini,
          'Entropy': roundSumEntropy}
resultDF = pd.DataFrame(output, 
                        columns = ['Variable','Gini impurity','Entropy'])
print("Result: \n", resultDF)

#print("list of Gini: ", roundSumGini)
#print("list of Entropy: ", roundSumEntropy)


# ##  Part 2: Building trees {-}
# 5. **Import the Boston using dataset using sklearn.datasets.**

# In[36]:


# Import the Boston using dataset
from sklearn import datasets
boston = datasets.load_boston()
#print(type(boston))
#print(boston.keys())
#print(boston.data.shape)
#print(boston.DESCR)

bosDF = pd.DataFrame(boston.data, columns = boston.feature_names) 
bosDF['MEDV'] = boston.target
#bosDF.head(3)
print("* Examine the null values: \n", bosDF.isnull().sum())

# Create a new column called “highPriced”
bosDF['highPriced'] = np.where(bosDF['MEDV']>35, 'Yes', 'No')
print("\n* Take a look of the dataset:\n", bosDF.head(3))
print("\n* Describe the independent variables:\n", bosDF.describe())
print("\n* Describe the target variable:\n", bosDF['highPriced'].describe())


# There are 506 observations in the dataset. About 90.5% (458/506) observations' median values are not greater that $35k. Only about 10% observations can be called "high-priced" houses.
# 
# 
# 6. **Using your helper functions and with highPriced as your output, find what the best split is along the AGE variable using each of Gini impurity and entropy as the splitting criterion.** (Hint: it may help to build this process as a function because you will be doing it many times throughout this problem set.) Assume the “left” side of each split contains all values less than and the “right” side contains all values equal to or greater than. What is the optimal split point when using Gini impurity? What about when using entropy? What if we calculate the same using the CRIM variable? Comment on any similarities and differences.
# 

# In[41]:


def splitFunction(dataframe, feature, splitValue, target):
    
    leftDf = dataframe[dataframe[feature] < splitValue]
    leftYes = len(leftDf[leftDf[target] == "Yes"])
    leftNo = len(leftDf[leftDf[target] == "No"])
    leftCount = len(leftDf)
    
    rightDf = dataframe[dataframe[feature] >= splitValue]
    rightYes = len(rightDf[rightDf[target] == "Yes"])
    rightNo = len(rightDf[rightDf[target] == "No"])
    rightCount = len(rightDf)
    
    # return a list
    return [leftCount, leftYes, leftNo, rightCount, rightYes, rightNo]


# In[259]:


def bestFunction(df, columnName, targetName, valueList):
    sumGini = []
    sumEntropy = []
    tempResultGini = {}
    tempResultEntropy = {}
    for value in valueList:
        #print(value)
        gini = []
        entropy = []
        weights = []

        tempList = splitFunction(df, columnName, value, targetName)
        #print("tempList:", tempList)

        # count=0 means we didn't split the dataset.
        if ((tempList[0]!=0) | (tempList[3]!=0)):
            gini.append(calcGini(tempList[1], tempList[2]))
            gini.append(calcGini(tempList[4], tempList[5]))
            weights.append(tempList[0]/(tempList[0]+tempList[3]))
            weights.append(tempList[3]/(tempList[0]+tempList[3]))
            entropy.append(calcEntropy(tempList[1], tempList[2]))
            entropy.append(calcEntropy(tempList[4], tempList[5]))
            #print("calcGini Left:", calcGini(tempList[1], tempList[2]))
            #print("calcGini Right:", calcGini(tempList[4], tempList[5]))
        #sumGini.append(weightedSum(gini, weights))    
        tempResultGini[value] = weightedSum(gini, weights)
        tempResultEntropy[value] = weightedSum(entropy, weights)
    #print(sorted(tempResultEntropy, key=tempResultEntropy.__getitem__)[0:5])
    thresholdGini = sorted(tempResultGini, key=tempResultGini.__getitem__)[0]
    finGini = tempResultGini[sorted(tempResultGini, key=tempResultGini.__getitem__)[0]]
    thresholdEntropy = sorted(tempResultEntropy, key=tempResultEntropy.__getitem__)[0]
    finEntropy = tempResultEntropy[sorted(tempResultEntropy, key=tempResultEntropy.__getitem__)[0]]
    #print(thresholdGini, round(finGini,3)) 
    #print(thresholdEntropy, round(finEntropy,3)) 
    
    output = {'Threshold': [thresholdGini, thresholdEntropy],
             "Value": [round(finGini,3), round(finEntropy,3)]}
    df = pd.DataFrame(data=output, index=["Gini impurity", "Entropy"])
    print(df)


# In[260]:


# the best split along the AGE variable
print("the best split along the variable(AGE)")
bestFunction(bosDF, "AGE", 'highPriced', bosDF["AGE"])


# We have the same thresholds when using both Gini impurity and entropy. It means both two method of making splits get the same answer in this case.

# In[261]:


# the best split along the CRIM variable
print("the best split along the variable(CRIM)")
bestFunction(bosDF, "CRIM", 'highPriced', bosDF["CRIM"])


# We have the different thresholds when using Gini impurity and entropy. It means two method of making splits cannot get the same answer in this case. We may have different optimal split points when we use different method to make splits with the same variables.

# 7. **Import sklearn’s DecisionTreeClassifier and find what the optimal split is along the AGE variable using entropy.**

# In[283]:


from sklearn import tree
from sklearn.model_selection import cross_val_score

dtree = tree.DecisionTreeClassifier(criterion='entropy')
dtree.fit(bosDF[['AGE']],bosDF[['highPriced']])

#print(type(tree.export_graphviz(dtree, label='root)')))
#print((tree.export_graphviz(dtree, label='root')))

#print(dtree.tree_.children_left[0]) #array of left children
#print(dtree.tree_.children_right[0])#array of right children
#print(dtree.tree_.feature[0]) #array of nodes splitting feature
print("Optimal point:", dtree.tree_.threshold[0])#array of nodes splitting points
#print(dtree.tree_.value[0]) #array of nodes values

#print("Entropy:", dtree.tree_.feature) 


# They are quite different. The threshold in the model I built is 37.3 while it in the sklearn model is 37.25. That may because I looked at axis-aligned splits at every data point, but sklearn looks at axis-aligned splits between every pair of data points. Thus, below I would like to make splits ar the pairs of data points.

# In[266]:


# the best split along the AGE variable
#bestFunction(bosDF, "AGE", 'highPriced')

# compose the pairs
pairValue = 0
pairList = []
for value1 in bosDF["AGE"]:
    for value2 in bosDF["AGE"]:
        pairValue = (value1+value2)/2
        #print(pairValue, value1, value2)
        pairList.append(pairValue)
pairList = sorted(pd.unique(pairList).tolist())
bestFunction(bosDF, "AGE", 'highPriced', pairList)


# After making the splits at the pairs of data points, I had the same threshold with the sklearn model.

# 8. **Using the RM, LSTAT, and RAD variables, build a a decision tree with 2 levels (3 split points, 4 leaf nodes) based on entropy.**

# In[110]:


dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
dtree.fit(bosDF[["RM", "LSTAT", "RAD"]], bosDF[['highPriced']])

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=["RM", "LSTAT", "RAD"],class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dtreeQ8.png')
Image(graph.create_png())


# In[129]:


# Show the splits as a pandas DataFrame
#print(tree.export_graphviz(dtree, label='root'))
output = {'level': [1,2,2],
          'col': ["RM", "LSTAT", "RM"],
          'threshold': dtree.tree_.threshold[[0,1,4]]}
splitOutput = pd.DataFrame(output, columns = ["level", 'col', 'threshold'])
print(splitOutput)


# 9. **Visualize your splits on an X-Y plane.**

# In[136]:


import matplotlib.pyplot as plt

plt.scatter(bosDF[["RM"]], bosDF[["LSTAT"]], color='#FF5126', lw=0.01)
plt.title("splits for my 2-level decision tree")
plt.xlabel("RM")
plt.ylabel("LSTAT")

# add line
plt.axvline(x=splitOutput['threshold'][0], label='first level', color='#162C9B', linestyle='--', linewidth=3)
plt.axhline(y=splitOutput['threshold'][1], xmin=0, xmax=(splitOutput['threshold'][0]-3)/(9.2-3), 
            label='second level-left',color='#37DC94', linestyle='--', linewidth=3)
plt.axvline(x=splitOutput['threshold'][2], label='second level-right',color='#228B22', linestyle='--', linewidth=3)
#plt.axvline(x=5, ymin=0.25, ymax=0.75)
#ax2.spines["top"].set_alpha(.3)
plt.legend()
plt.show()


# The blue line can show that the first split made more observations (458) on the left side while only 48 observations on the right side. The light-green line represented the second level split of the left side while the forestgreen line represented the second level split of the right side. The variance is higher on the left side.
# 
# 
# ##  Part 3: Making predictions {-}
# 10. **Create a training/test split for your data.**

# In[140]:


# split data
testIndex = []
trainingIndex = []

for i in range(len(bosDF)):
    if(i%5==0):
        testIndex.append(i)
    else:
        trainingIndex.append(i)
#print(testIndex)        
#print(trainingIndex)

# the counts of observations
testDF = bosDF.loc[testIndex]
print(len(testDF), "observations are in the test dataset.")
trainingDF = bosDF.loc[trainingIndex]
print(len(trainingDF), "observations are in the training dataset.")

# How does the target variable look across the two datasets?
testTargetDF = pd.DataFrame(testDF["CRIM"].groupby(testDF["highPriced"]).count())
testTargetDF["Percentage"] = round(testTargetDF["CRIM"]/len(testDF) * 100, 2)
testTargetDF.rename(columns={"CRIM": "Count"}, inplace=True)

trainingTargetDF = pd.DataFrame(trainingDF["CRIM"].groupby(trainingDF["highPriced"]).count())
trainingTargetDF["Percentage"] = round(trainingTargetDF["CRIM"]/len(trainingDF) * 100, 2)
trainingTargetDF.rename(columns={"CRIM": "Count"}, inplace=True)

print("\nthe target variable's distribution in the test dataset \n", testTargetDF)
print("\nthe target variable's distribution in the training dataset \n", trainingTargetDF)


# The percentage of classification "No" in the training dataset is 89.85% while it is 93.14% in the test dataset.
# The percentage of classification "Yes" in the training dataset is 10.15% while it is 6.86% in the test dataset.

# 11. **Create a baseline set of predictions.**

# In[225]:


# The performance of prediction
def measurePredict(target, predict):
    #Accuracy
    from sklearn import metrics
    print('Accuracy Score:', round(metrics.accuracy_score(target, predict), 3))
    
    confusionMatrix = pd.DataFrame({'predict': predict, 'realValue': target.values})
    print("\nConfusion Matrix:\n", confusionMatrix.groupby(['predict','realValue']).size())
    
    # confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(target, predict)
    #print("\nConfusion Matrix:\n", cm)

    sensitivity = round(cm[0,0]/(cm[0,0]+cm[0,1]), 3)
    print('\nSensitivity (true "No" rate):', sensitivity)

    specificity = round(cm[1,1]/(cm[1,0]+cm[1,1]), 3)
    print('Specificity (true "Yes" rate):', specificity)


# In[226]:


#Predict the response for test dataset
testPredict = []

for i in range(len(testDF)):
    randomSelectDF = (trainingDF[['highPriced']].sample(n=1))
    #print("randomSelectDF\n", randomSelectDF, "\n")
    randomSelect = randomSelectDF.iloc[0,0]
    #print("randomSelect:", randomSelect)
    testPredict.append(randomSelect)
#print(type(testPredict))

# Performance Measures for the Prediction Model
measurePredict(testDF.iloc[:,14],testPredict)


# In the training dataset, the percentage of classification "No" in the training dataset is 89.85%, and that means if randomly sampling from the variable "highPriced", we probably have the percentage of the prediction in the classification "No" in the test dataset is also around 89.85% while the percentage of real "No" cases in the test data is 93.14%. 
# In this case, the accuracy is 85.3%. 
# The sensitivity is 90.5%, and it represented that 90.5% of real non-highpriced observations are predicted positive (classification "No") in the test dataset. 
# The specificity is 14.3%, and it represented that 14.3% of real highpriced observations are predicted negative (classification "Yes").

# 12. **Now, use a 2-level decision tree to make predictions.**

# In[228]:



dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)

dtree = dtree.fit(trainingDF[["RM", "LSTAT"]], trainingDF[['highPriced']])

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=["RM", "LSTAT"],class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dtreeQ12.png')
Image(graph.create_png())


# The decision nodes are the same while they have different thresholds and different values of entropy in each decision node.

# In[230]:


#Predict the response for test dataset

testPredict = dtree.predict(testDF[["RM", "LSTAT"]])

# Performance Measures for the Prediction Model
measurePredict(testDF.iloc[:,14],testPredict)


# The accuracy is 97.1% and is higher than in the baseline. 
# The sensitivity is 100%, and it represented that all of real non-highpriced observations are predicted positive (classification "No") in the test dataset. 
# The specificity is 57.1%, and it represented that 57.1% of real highpriced observations are predicted negative (classification "Yes").
# Both the sensitivity and specificity are higher than in the baseline, so the predictions are better here.

# ##  Part 4: Comparing to out-of-the-box classifiers {-}
# 13. **Use sklearn’s DecisionTreeClassifier to recreate the decision tree that is trained on the training data.**

# In[231]:


# recreate the decision tree
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)

dtree = dtree.fit(trainingDF.iloc[:,0:13], trainingDF[['highPriced']])

colName = list(trainingDF.columns.values)
#print(colName)
del colName[-2:]
#print(colName)

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=colName,class_names=['No','Yes'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('dtreeQ13.png')
Image(graph.create_png())


# In[232]:


#Predict the response for test dataset
from sklearn.model_selection import cross_val_score
testPredict = dtree.predict(testDF.iloc[:,0:13])
#print(type(targetPredict))

# Performance Measures for the Prediction Model
measurePredict(testDF.iloc[:,14],testPredict)


# The decision tree is the same as the decision tree in Q12. It showed that "RM" and "LSTAT" are the main features.
# The predictions are also the same as them in Q12 because of the same decision tree. 
# The accuracy, the sensitivity and the specificity are higher than in the baseline, so the predictions are better here when compared to the baseline.

# 14. **Use sklearn’s BaggingClassifier to create a bagging classifier whose base is a DecisionTreeClassifier with 2 levels and entropy as the split criterion.**

# In[234]:


from sklearn.ensemble import BaggingClassifier
dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
bag = BaggingClassifier(base_estimator = dtree)
bagFit = bag.fit(trainingDF.iloc[:,0:13], trainingDF[['highPriced']])

#Predict the response for test dataset
testPredict = bagFit.predict(testDF.iloc[:,0:13])
#print(type(targetPredict))

# Performance Measures for the Prediction Model
measurePredict(testDF.iloc[:,14],testPredict)


# The accuracy is 97.1% and is the same as it in the Q13.
# The sensitivity is 98.9%, and it represented that 98.9% of real non-highpriced observations are predicted positive (classification "No") in the test dataset. It is lower than in the Q13. 
# The specificity is 71.4%, and it represented that 71.4% of real highpriced observations are predicted negative (classification "Yes"). It is higher than in the Q13. 
# To show which classifier is better, we should know which performance indicator (sensitivity or specificity) is more important based on different conditions.

# 15. **Use sklearn’s RandomForestClassifier to create a random forest classifier whose base is a decision tree with 2 levels and entropy as the split criterion.**

# In[238]:


from sklearn.ensemble import RandomForestClassifier
#dtree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
RF = RandomForestClassifier(criterion='entropy', max_depth=2)
RFFit = RF.fit(trainingDF.iloc[:,0:13], trainingDF[['highPriced']])

#Predict the response for test dataset
testPredict = bagFit.predict(testDF.iloc[:,0:13])
#print(type(targetPredict))

# Performance Measures for the Prediction Model
measurePredict(testDF.iloc[:,14],testPredict)


# The confusion matrix of the predictions is the same as Q14 which means the performance of predictions is also the same as Q14. Both the random forest classifier and the bagging classifier can have the similar performances in the dataset.
# 
# Both the random forest classifier and the decision tree classifier have the same accuracy. The sensitivity here is lower than in the Q13 while the specificity here is higher than in the Q13.
