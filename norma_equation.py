import numpy as np
class Normal_Equation():
	"""docstring for ClassName"""
	def __init__(self):
		self.about = "Normal equation"
		self.W=[]

	def evaluate(self,x,y):
		m = x.shape[0]
		one=np.ones((m,1))
		complete=np.hstack((one,x))
		self.W = np.linalg.inv(complete.T.dot(complete)).dot(complete.T).dot(y)
		return self.W
	def predict (self, target):
		m = target.shape[0]
		one = np.ones((m,1))
		return np.hstack((one,target)).dot(self.W)