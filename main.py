import numpy as np
from norma_equation import Normal_Equation
from gradient import Gradient_Descent
import corr_checking
ne = Normal_Equation()
gd = Gradient_Descent()


######
try:
	f_data=open('./data_set.csv','r') #set for eduaction with excluded the last twenty parameters
	f_target=open('./target.csv','r') 
	check_data=open('./data_set_check.csv','r') # full setl for checking how our model is working
	target_data_check=open('./target_check.csv','r')
except FileNotFoundError:
	print ("File not found")
	exit()




target=np.loadtxt(f_target,delimiter=',',skiprows=1,usecols=1)
data_for_learning = np.loadtxt(f_data,delimiter=',',skiprows=1,usecols=(1,2,4))
data_set_for_check = np.loadtxt(check_data,delimiter=',',skiprows=1,usecols=(1,2,4))
target_data_for_check = np.loadtxt(target_data_check,delimiter=',',skiprows=1,usecols=1)


ne.evaluate(data_for_learning,target)
ne_res = ne.predict(data_set_for_check)
ne_error = np.sum(abs(ne_res - target_data_for_check))/ne_res.shape[0]


gd.fit(data_for_learning,target,learning_rate=0.001,nsteps=10000000,weight_high=100000, weight_low=0)
gd_res = gd.predict(data_set_for_check).reshape((data_set_for_check.shape[0],))
gd_error = np.sum(abs(gd_res- target_data_for_check))/gd_res.shape[0]

print ('Normal equation:')
print (ne_res)
print ("mean absolute error for normal equation: ",ne_error)
print ("Gradient_Descent:")
print(gd_res)
print ("mean absolute error for gradient descent: ",gd_error)