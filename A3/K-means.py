''' 
# K-Means Clustering Scratch Implementation
#
# 
# Code Author: Kunal Vinay Kumar Suthar
# Assignment 3 for CSE:575 Statistical Machine Learning
# ASU ID: 1215112535
# 
'''
import numpy as np
import pandas as pd
import random
from random import shuffle
import math
from collections import Counter
import matplotlib.pyplot as plt
import operator

def main():

	print("Helllo")
	data=pd.read_csv('breast-cancer-wisconsin.data', sep=",", header=None)
	print(data)
	data=data.values[:,1:10]
	print(data)




if __name__=="__main__":
	main()



