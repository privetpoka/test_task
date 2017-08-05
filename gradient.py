import numpy as np
class LinRegression:
    '''Linear Regression.'''
    def __init__(self):
        self.about = "Gradient descent"
        self.W = [] # model's weights
        self.is_scalling = False # is feature scaling used
        
    def cost_function(self, y_real, y_pred): 
        # cost function for gradient descent algorithm
        return np.sum(abs(y_pred-y_real))/(len(y_real))
    
    def gradient_descent_step(self, learning_rate, dy, m, n, X_tr):
        # one gradient descent step
        s = (np.dot(dy.T, X_tr)).reshape(n, 1)
        dW = 2*(learning_rate*s/m).reshape(n, 1)
        return self.W - dW
    
    def normalize(self, X):
        # normilize X table
        for j in range(X.shape[1]):
            X[:,j] = X[:,j]/np.max(X[:,j])
        return X
    
    def fit(self, X, y, learning_rate = 0.99, nsteps = 3000, e = 0.000000001,
            weight_low = 0, weight_high = 1,
            is_scalling = False):
        # train our Linear Regression model
        
        np.random.seed(0)
        X = X.astype(float)
        
        # Normilize process
        if is_scalling == True:
            X = self.normalize(X)
            self.is_scalling = True
        m = X.shape[0]
        # add one's column to X
        X = np.hstack( (np.ones(m).reshape(m, 1), X) )
        n = X.shape[1]
        
        # Weights: random initialization
        self.W = np.random.randint(low = weight_low, high = weight_high, size=(n, 1))
            
        y_pred = np.dot(X, self.W)
        cost0 = self.cost_function(y, y_pred)
        y = y.reshape(m, 1)
        k = 0
        
        ########## Gradient descent's steps #########
        while True:
            dy = y_pred - y
            W_tmp = self.W
            self.W = self.gradient_descent_step(learning_rate, dy, m, n, X)
            y_pred = np.dot(X, self.W)
            cost1 = self.cost_function(y, y_pred)
            k += 1
            if (cost1 > cost0):
                self.W = W_tmp
                break    
                
            if ((cost0 - cost1) < e) or (k == nsteps):
                break
                
            cost0 = cost1
        #############################################
        return self.W # return model's weights
    
    def predict(self, X):
        m = X.shape[0]
        if self.is_scalling == False:
            return np.dot( np.hstack( (np.ones(m).reshape(m, 1),
                                       X.astype(float)) ) ,
                          self.W)
        else:
            return np.dot( np.hstack( (np.ones(m).reshape(m, 1),
                                       self.normalize(X.astype(float))) ),
                          self.W)

f = open ('./data_set.csv','r')
x =  np.loadtxt(f,skiprows=1,delimiter=',',usecols=(1,2,4))
g = open ('target.csv','r')
y =  np.loadtxt(g,skiprows=1,delimiter=',',usecols=1)
lr = LinRegression()
lr.fit(x,y,learning_rate=0.0001,nsteps=10000)
print(lr.W)
f = open ('./data_set_check.csv','r')
data_set_for_check = np.loadtxt(f,delimiter=',',skiprows=1,usecols=(1,2,4))
print (lr.predict(data_set_for_check))
