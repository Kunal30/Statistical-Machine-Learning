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
from sklearn.metrics.pairwise import euclidean_distances

def main():
	
	# Extracting 60000 and 10000 MNIST images
	(trX, trY), (tsX, tsY)= mnist.load_data()	
	
	# Reducing them to 6000 and 1000 respectively
	trX=trX[0:6000]
	trY=trY[0:6000]
	tsX=tsX[0:1000]
	tsY=tsY[0:1000]
	
	#Flattening the images from 28*28 to 784
	trX=np.reshape(trX,(6000,784))
	
	tsX=np.reshape(tsX,(1000,784))
	
	

	
	#calculating Euclidean distance between each
	#test point to training point
	tsX_to_trX_d=calc_distance(tsX,trX)
	trX_to_trX_d=calc_distance(trX,trX)
	tsX_to_trX_d= euclidean_distances(tsX,trX)
	trX_to_trX_d= euclidean_distances(trX,trX)
	
	
	K=[1,9,19,29,39,49,59,69,79,89,99]
	results_test=[]
	results_train=[]
	
	for i in K:
		result_ts=0
		result_tr=0
		for j in range(1,6):
			print("Iteration ",j)
			result_ts=result_ts+apply_KNN(tsX_to_trX_d,trY,tsY,i)
			result_tr=result_tr+apply_KNN(trX_to_trX_d,trY,trY,i)
		results_test.append(100-(result_ts/5))
		results_train.append(100-(result_tr/5))		
	
	print(results_test)	
	print(results_train)

	plotgraph(results_train,results_test,K)

def plotgraph(res_tr,res_ts,K):
	fig= plt.figure()
	plt.title('KNN')
	plt.plot(K,res_tr,'r',label='Training Error')
	plt.plot(K,res_ts,'b',label='Testing Error')	
	plt.ylabel('Error %')
	plt.xlabel('Value of K')
	plt.legend()		
	plt.show()

		
def apply_KNN(tsX_to_trX_d,trY,tsY,K):

	Y_pred=np.zeros((tsY.shape))
	
	for i in range(0,tsY.shape[0]):
		distances=tsX_to_trX_d[i]
		labels= trY		
		distances,labels= zip(*sorted(zip(distances,labels)))				
		labels=np.asarray(labels)		
		distances=distances[0:K]
		labels=labels[0:K]		
		Knn_dict=Counter(labels)
		predicted_value=max(Knn_dict.items(), key=operator.itemgetter(1))[0]
		Y_pred[i]=int(predicted_value)
	accuracy= (np.sum(tsY == Y_pred, axis=0)*1.00) / tsY.shape[0]	
	print(accuracy*100,"%")	
	return accuracy*100

def calc_distance(tsX,trX):
	
	d=np.zeros((1000,6000))

	for i in range(0,1000):
		for j in range(0,6000):
			diff_sq=np.square(tsX[i]-trX[j])
			summation=np.sum(diff_sq,axis=0)
			print(i," ",j)			
			dist=np.sqrt(summation)	
			print(dist)		
			d[i][j]=dist

	return d	












if __name__=="__main__":
	main()