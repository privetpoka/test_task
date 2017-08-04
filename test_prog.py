from pandas import read_csv, DataFrame
import numpy as np
### here is dataset without the last 20 result and parameters
f_data=open('./data_set.csv','r')
dataset = read_csv('./data_set.csv')
#### here is we're checking parametres, which are corresponding between each other in order to exclude them
print (dataset.corr())
f_target=open('./target.csv','r') 
target=np.loadtxt(f_target,delimiter=',',skiprows=1,usecols=1)
data = np.loadtxt(f_data,delimiter=',',skiprows=1,usecols=(1,2,4))
### add column filled with ones
one=np.ones((345,1))
comlete=np.hstack((one,data))
### eveluaiting 
result=np.linalg.inv(data.T.dot(data)).dot(data.T).dot(target)
print(result)
print(result.shape)
check_data=open('./data_set_check.csv','r')
target_data_check=open('./target_check.csv','r')
data_set_for_check = np.loadtxt(check_data,delimiter=',',skiprows=1,usecols=(1,2,4))
target_data_for_check = np.loadtxt(target_data_check,delimiter=',',skiprows=1,usecols=1)
evaluated_result = data_set_for_check.dot(result)
print(evaluated_result)
print(target_data_for_check)
my_error = (abs(target_data_for_check - evaluated_result))
print (np.mean(target_data_for_check))
print(np.mean(my_error)/np.mean(target_data_for_check))