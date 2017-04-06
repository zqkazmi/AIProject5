
# Based on Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014
# Updated by Megan Olsen, 2017 -- includes all continuous variable code and validation and testing

# Code to run the decision tree on the Iris dataset

''' iris feature information
	             Min  Max   Mean
   sepal length: 4.3  7.9   5.84
    sepal width: 2.0  4.4   3.05
   petal length: 1.0  6.9   3.76
    petal width: 0.1  2.5   1.20
'''

import dtree_book
import numpy as np

#data is the full data to modify
#feature is the index of the feature to split on
#splitpoints is a list of each bin definition
#if there are 3 splitpoint values, then there are 3 bins
def fix_continuous_values(data,feature,splitpoints):
    for datapoint in data:
        is_set = False
        for i in range(len(splitpoints)):
            if datapoint[feature] < splitpoints[i]:
                datapoint[feature] = i
                is_set = True
                break
        if not is_set:
            datapoint[feature] = len(splitpoints)

#Change all feature values to be binned instead of continuous values
#For instance, if the bins are 1,2, then values <1 are given a value of 0, <2 are given 1, and >=2 are given value 2.
def continous_to_bins(data):
    #Second, set the bins for each feature
    setpoints0 = [5,6,7] #three bins for first feature
    setpoints1 = [3,4] #two bins for second feature
    setpoints2 = [6] #one bins for third feature
    setpoints3 = [1,2] #two bins for fourth feature
    #Three, apply the bins to each feature
    fix_continuous_values(iris,0,setpoints0)
    fix_continuous_values(iris,1,setpoints1)
    fix_continuous_values(iris,2,setpoints2)
    fix_continuous_values(iris,3,setpoints3)

#Read in the data
tree = dtree_book.dtree_book()
iris,classes,features = tree.read_data('iris_proc.data',",")

#Change continuous variables to be in a particular bin of values
#First convert everything to numbers instead of strings
iris_new = []
for i in range(len(iris)):
    iris_new.append(list(map(float,iris[i])))
iris = iris_new
continous_to_bins(iris) #call function to bin everything

#Print original data, then split between train, test, validate
print("The data:",iris)
print("The class for each datapoint:",classes)
print("The features:",features)
train = iris[::2]
traint = classes[::2]
valid = iris[1::4]
validt = classes[1::4]
test = iris[3::4]
testt = classes[3::4]

#Check for balanced dataset by counting how many instances of each class are in training set
zero = traint.count("0")
one = traint.count("1")
two = traint.count("2")
print(zero,one,two)

#Learn the tree
t=tree.make_tree(iris,classes,features)
tree.printTree(t,' ')

#Try to classify the test data
classification = tree.classifyAll(t,test)
print("Classification:\n",classification)
print("True Classes:")
print(testt)

#Confusion matrix
#row is actual class, column is classification
matrix = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(classification)):
    matrix[int(testt[i])][int(classification[i])] += 1
print("Confusion Matrix")
for row in matrix:
    print(row)

