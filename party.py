
# Code from Chapter 12 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014
# Updated by Megan Olsen, 2017

# Code to run the decision tree on the Party dataset
import dtree_book

#Learn the tree
tree = dtree_book.dtree_book()
party,classes,features = tree.read_data('party.dat',"\t")
print("The data:",party)
print("The class for each datapoint:",classes)
print("The features:",features)
t=tree.make_tree(party,classes,features)
tree.printTree(t,' ')

#Try to classify the original data to see accuracy
#Note that this is NOT a valid way to test accuracy, just a quick example
classification = tree.classifyAll(t,party)
print("Classification:\n",classification)
print("True Classes:")
print(classes)

if classes == classification:
    print("Perfect match!")
