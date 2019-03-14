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
from collections import Counter
import matplotlib.pyplot as plt
from keras.datasets import mnist
import operator

def main():
	
	# Extracting 60000 and 10000 MNIST images
	(trX, trY), (tsX, tsY)= mnist.load_data()	
	
	# Reducing them to 6000 and 1000 respectively
	trX=trX[0:6000]
	trY=trY[0:6000]
	tsX=tsX[0:1000]
	tsY=tsY[0:1000]
	
	#Flattening the images from 28*28 to 784
	trX=np.reshape(trX,(6000,784,1))
	# trY=np.reshape(trX,(784,1))
	tsX=np.reshape(tsX,(1000,784,1))
	# trX=np.reshape(trX,(784,1))
	

	print(tsY.shape[0])
	#calculating Euclidean distance between each
	#test point to training point
	tsX_to_trX_d=calc_distance(tsX,trX)

	# print(tsX_to_trX_d.shape)
	K=[99]
	results=[]
	
	for i in K:
		result=0
		for j in range(1,6):
			result+=apply_KNN(tsX_to_trX_d,trY,tsY,i)
		results.append(result/5)	

def apply_KNN(tsX_to_trX_d,trY,tsY,K):

	Y_pred=np.zeros((tsY.shape))
	for i in range(0,tsY.shape[0]):
		distances=tsX_to_trX_d[i]
		labels= trY
		distances,labels= zip(*sorted(zip(distances,labels)))
		# print(distances)
		# print("gapppp")
		distances=distances[0:K]
		labels=labels[0:K]
		Knn_dict=Counter(labels)
		predicted_value=max(Knn_dict.items(), key=operator.itemgetter(1))[0]
		Y_pred[i]=int(predicted_value)
		# print(Knn_dict)
		# print(predicted_value)		
		# # print(distances)
		# print(labels)
	print(tsY)
	print("Gap")
	print(Y_pred)	
	accuracy= (np.sum(tsY == Y_pred, axis=0)*1.00) / 1000	
	print(accuracy*100,"%")	
def calc_distance(tsX,trX):
	d=np.zeros((1000,6000))

	for i in range(0,1000):
		for j in range(0,6000):
			# print(tsX.shape)
			# print(trX.shape)
			diff_sq=np.square(tsX[i]-trX[j])
			summation=np.sum(diff_sq,axis=0)
			print(i," ",j)			
			dist=np.sqrt(summation)			
			d[i][j]=dist

	return d		
# def apply_KNN(trX,trY,tsX,tsY,K):














if __name__=="__main__":
	main()