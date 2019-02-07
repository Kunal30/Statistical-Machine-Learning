import numpy as np
import random
from random import shuffle
import math
import collections
import matplotlib.pyplot as plt

def main():
	
	#Extracting features and Labels from the file
	X,Y= extract_data('data_banknote_authentication.txt') 
	
	#Now performing 3 fold cross validation on the dataset
    #to get the get the data for the 6 fractions over 5 runs
	
	train_test_data,frac_values_per_dataset= three_fold_cross_validation(X,Y,[0.01,0.02,0.05,0.1,0.625,1]) 
	print(train_test_data)
	#Now we can train the Gaussian Naive Bayes classifier
	#for all training set sizes
	hash_map_accuracies={.01: [], 0.02: [], 0.05: [], 0.1: [], 0.625: [], 1:[] } 

	
	for i in range(0,train_test_data.shape[0]):
		X_train=train_test_data[i][0]
		Y_train=train_test_data[i][1]
		X_test=train_test_data[i][2]
		Y_test=train_test_data[i][3]				
		
		parameters= train_GNB_classifier(X_train,Y_train)
		predictions= predict_GNB_classifier(X_test,parameters)
		accuracy= calc_accuracy(predictions,Y_test)
		hash_map_accuracies[frac_values_per_dataset[i]].append(accuracy)
	
	learning_curve_plot1={}
	
	for i in hash_map_accuracies.items():		  
		  learning_curve_plot1[i[0]]=np.mean(i[1])

	print(learning_curve_plot1)	  
	
	#Now plotting the curve for accuracy vs training-set fraction
	# plot_graph(learning_curve_plot,'GNB')	  

	#Now switching to Logistic Regression

	hash_map_accuracies={.01: [], 0.02: [], 0.05: [], 0.1: [], 0.625: [], 1:[] }
	
	for i in range(0,train_test_data.shape[0]):
		X_train=train_test_data[i][0]
		Y_train=train_test_data[i][1]
		X_test=train_test_data[i][2]
		Y_test=train_test_data[i][3]				
		
		W,b= train_LR_classifier(X_train,Y_train,0.05)
		predictions= predict_LR_classifier(X_test,W,b)
		accuracy= calc_accuracy(predictions,Y_test)
		hash_map_accuracies[frac_values_per_dataset[i]].append(accuracy)
	
	learning_curve_plot2={}
	
	for i in hash_map_accuracies.items():		  
		  learning_curve_plot2[i[0]]=np.mean(i[1])
	print(learning_curve_plot2)	  
	plot_graph(learning_curve_plot1,learning_curve_plot2,'GNB vs LR')

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def plot_graph(plot1,plot2,title):
	
	X,Y1= zip(*sorted(plot1.items()))
	X,Y2= zip(*sorted(plot2.items()))
	fig= plt.figure()
	plt.title(title)
	plt.plot(X,Y1,'r',label='GNB')
	plt.plot(X,Y2,'b',label='LR')
	# plt.yticks(np.arange(min(Y1),1, step=0.1))
	plt.ylabel('Accuracy')
	plt.xlabel('Training Data Size')
	plt.legend()		
	plt.show()
	

def train_LR_classifier(X_train, Y_train, eta):	
	X_train=X_train.T
	Y_train=Y_train.T

	rows,cols=X_train.shape

	W=np.zeros((rows,1))
	b=0
	for i in range(0,10):
		Z=np.dot(W.T,X_train)+b
		Y_hat=sigmoid(Z)
		loss=Y_train-Y_hat
		db=np.sum(loss)/cols
		dW=np.matmul(X_train,loss.T)/cols
		W=W+eta*dW
		b=b+eta*db

	return W,b

def train_GNB_classifier(X_train, Y_train):

	#calculating priors and parameters(u and sigma_sqr)
	parameters={}

	number_of_diffnums= collections.Counter(Y_train)

	parameters['Prior_y_1']= number_of_diffnums[1]*1.00/len(Y_train)
	parameters['Prior_y_0']= number_of_diffnums[0]*1.00/len(Y_train)

	ones= X_train[Y_train[:]==1.00]
	zeros= X_train[Y_train[:]==0.00]
	
	parameters['u_1']=np.mean(ones,axis=0)
	parameters['u_0']=np.mean(zeros,axis=0)
	parameters['var_1']=np.var(ones,axis=0)
	parameters['var_0']=np.var(zeros,axis=0)

	return parameters

def predict_LR_classifier(X_test,W,b):
	predictions=[]
	X_test=X_test.T
	Z=np.dot(W.T,X_test)+b
	Y_hat=sigmoid(Z)
	
	Y_hat=Y_hat.T
	
	for i in range(0,X_test.shape[1]):
		if Y_hat[i]>=0.5:
			predictions.append(1)
		else:
			predictions.append(0)	
			
	return predictions
def predict_GNB_classifier(X_test, parameters):
	
	predictions=[]

	#We will calculate likelihood and posteriors in this section
	#Here we assume that all the features xi are independent

	likelihood_x_y_1= (np.exp(-np.square(X_test-parameters['u_1'])/(2*parameters['var_1'])))/(np.sqrt(2*3.14*parameters['var_1']))
	likelihood_x_y_0= (np.exp(-np.square(X_test-parameters['u_0'])/(2*parameters['var_0'])))/(np.sqrt(2*3.14*parameters['var_0']))

	# print(likelihood_x_y_0.shape)
	#Now because of conditional independence
	likelihood_x1_x2_x3_x4_y_1= likelihood_x_y_1[:,0]*likelihood_x_y_1[:,1]*likelihood_x_y_1[:,2]*likelihood_x_y_1[:,3]
	likelihood_x1_x2_x3_x4_y_0= likelihood_x_y_0[:,0]*likelihood_x_y_0[:,1]*likelihood_x_y_0[:,2]*likelihood_x_y_0[:,3]

	Posterior_Y_X_1= likelihood_x1_x2_x3_x4_y_1*parameters['Prior_y_1']
	Posterior_Y_X_0= likelihood_x1_x2_x3_x4_y_0*parameters['Prior_y_0']

	# print(Posterior_Y_X_0.shape)

	for i in range(0,457):
		if Posterior_Y_X_1[i] >= Posterior_Y_X_0[i]:
			predictions.append(1)
		else:
			predictions.append(0)	
	
	# predictions[Posterior_Y_X_1 >= Posterior_Y_X_0]=1
	# predictions[Posterior_Y_X_1 < Posterior_Y_X_0]=0
	predictions=np.asarray(predictions)
	# print(predictions)
	return predictions

def calc_accuracy(predictions, Y_test):
	accuracy_matrix=np.zeros(len(Y_test))
	accuracy_matrix[predictions == Y_test]=1
	accuracy_matrix[predictions != Y_test]=0

	return (((sum(accuracy_matrix))*1.00)/len(Y_test))


def three_fold_cross_validation(X,Y,fractions):

	train_test_data=[]
	
	# 1372 is not divisible by 3
	# therefore 3 fold cannot be applied directly
	X=X[:-(len(Y)%3)]
	Y=Y[:-(len(Y)%3)]
	# X=X.tolist()
	# Y=Y.tolist()
	
	frac_values_per_dataset=[]	

	for fraction in fractions:		
		#randomization starts here
		state=random.getstate()
		X=random.sample(X,X.shape[0])
		random.setstate(state)
		Y=random.sample(Y,Y.shape[0])
		# state = np.random.get_state()
		# X=np.take(X,np.random.permutation(X.shape[0]),axis=0,out=X)
		# np.random.set_state(state)
		# Y=np.take(Y,np.random.permutation(Y.shape[0]),axis=0,out=Y)
		#randomization ends here
		X=np.asarray(X)
		Y=np.asarray(Y)
		#now splitting the data into 3 folds
		X_temp=np.split(X,3)
		Y_temp=np.split(Y,3)

		# now generating train and test data from the folds
		for i in range(0,3):
			X_TEST = X_temp[i]
			Y_TEST = Y_temp[i]
			# a=X_temp[:i]
			# b=X_temp[i+1:]
			# a=np.asarray(a)
			# b=np.asarray(b)
			# print(a.shape)
			# print(b.shape)
			X_TRAIN = np.concatenate(X_temp[:i]+X_temp[i+1:],axis=0)
			Y_TRAIN = np.concatenate(Y_temp[:i]+Y_temp[i+1:],axis=0)

			#Collecting data over 5 runs
			for i in range(1,6):
				value=math.floor(fraction*Y_TRAIN.shape[0])
				frac_values_per_dataset.append(fraction)
				# print(X_TRAIN.shape)				
				# X_TRAIN_TEMP= X_TRAIN[:int(value)]
				# Y_TRAIN_TEMP= Y_TRAIN[:int(value)]
				state=random.getstate()
				X_TRAIN_TEMP=random.sample(X_TRAIN,int(value))
				random.setstate(state)
				Y_TRAIN_TEMP=random.sample(Y_TRAIN,int(value))
				X_TRAIN_TEMP=np.asarray(X_TRAIN_TEMP)
				Y_TRAIN_TEMP=np.asarray(Y_TRAIN_TEMP)
				train_test_data.append([X_TRAIN_TEMP,Y_TRAIN_TEMP,X_TEST,Y_TEST])

				#now randomizing the training data and label
				# X_TRAIN=random.sample(X_TRAIN,X_TRAIN.shape[0])
				# Y_TRAIN=random.sample(Y_TRAIN,Y_TRAIN.shape[0])
				# X_TRAIN=np.asarray(X_TRAIN)
				# Y_TRAIN=np.asarray(Y_TRAIN)
				# state = np.random.get_state()
				# X_TRAIN=np.take(X_TRAIN,np.random.permutation(X_TRAIN.shape[0]),axis=0,out=X_TRAIN)
				# np.random.set_state(state)				
				# Y_TRAIN=np.take(Y_TRAIN,np.random.permutation(Y_TRAIN.shape[0]),axis=0,out=Y_TRAIN)

	return np.array(train_test_data),frac_values_per_dataset

def extract_data(location):
	
	x1=[]
	x2=[]
	x3=[]
	x4=[]
	X=[]
	y=[]
	with open(location) as f:
		for line in f:
			data = line.split(',')
			x1.append(float(data[0]))
			x2.append(float(data[1]))
			x3.append(float(data[2]))
			x4.append(float(data[3]))
			y.append(float(data[4]))

	X.append(x1)		
	X.append(x2)		
	X.append(x3)		
	X.append(x4)		
	
	X=np.asarray(X)
	y=np.asarray(y)
	
	X=X.T
		
	return X,y		




if __name__ == "__main__":
    main()	