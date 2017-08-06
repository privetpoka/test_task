from pandas import read_csv, DataFrame

data_set = read_csv('./data_set.csv')
print ("Here is we're checking parametres, which are corresponding between each other in order to exclude them:")
print (data_set.corr())