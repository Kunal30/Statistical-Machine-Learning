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

	#Extracting the data from the file	
	data=pd.read_csv('bc.txt', sep=",", header=None)	
	data=data.values[:,1:10]
	K=[8]
	potential=[]
	print(data.shape)
	for i in K:
		potential.append(apply_KMeans(i,data))



def apply_KMeans(K,data):

	#assigning random initial centroids

	rand_k=select_random_centroid_indices(K)	
	
	prev_centroids=None
	cluster_class=None
	new_centroids=data[rand_k]
	
	for i in range(0,1):
		print("Iteration---------------->",i)
		cluster_class=reassign_centroid(new_centroids,data)
		prev_centroids=new_centroids
		new_centroids=recompute_new_centroids(K,cluster_class,data)

	potential=compute_PotentialFunction(new_centroids,data,cluster_class)


def compute_PotentialFunction(new_centroids,data,cluster_class):
	


# Euclidean Distance 
def distance(a, b):
    return np.linalg.norm(a - b,axis=1)

def recompute_new_centroids(K,cluster_class,data):

	centroids=[]
	for i in range(0,K):
		sum_t=0
		points=0
		for j in range(0,699):
			if(cluster_class[j]==i):
				sum_t=sum_t+data[j]
				points=points+1
		centroids.append(sum_t*1.0/points)
	
	centroids=np.asarray(centroids)

	return centroids			

def reassign_centroid(new_centroids,data):

	cluster_class=[]
	
	for i in range(0,699):
		distances=distance(data[i],new_centroids)
		cluster=np.argmin(distances)
		cluster_class.append(cluster)

	# print(np.asarray(cluster_class).shape)	
	cluster_class=np.asarray(cluster_class)
	return cluster_class	


def select_random_centroid_indices(K):
	rand_k=random.sample(np.arange(0,700,1),K)
	return rand_k


if __name__=="__main__":
	main()



