import pandas as pd
import numpy as np


def main():
	
	X,Y= extract_data('data_banknote_authentication.txt') #Extracting features and Labels from the file
	


def three_fold_cross_validation(X,Y):


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