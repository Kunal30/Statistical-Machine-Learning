''' 
# K-Nearest Neighbors Scratch Implementation
#
# 
# Code Author: Kunal Vinay Kumar Suthar
# Assignment 2 for CSE:575 Statistical Machine Learning
# ASU ID: 1215112535
# Used Keras' library to automatically get the MNIST dataset as mentioned in the problem statement
'''
import numpy as np
import random
from random import shuffle
import math
import collections
import matplotlib.pyplot as plt
from keras.datasets import mnist

def main():
	
	# Extracting 60000 and 10000 MNIST images
	(trX, trY), (tsX, tsY)= mnist.load_data()	
	
	# Reducing them to 6000 and 1000 respectively
	trX=trX[0:6000]
	trY=trY[0:6000]
	tsX=tsX[0:1000]
	tsY=tsY[0:1000]
	




















if __name__=="__main__":
	main()