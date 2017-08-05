import numpy as np
class Gradient_Descent:
    '''Linear Regression.'''
    def __init__(self):
        self.about = "Gradient descent"
        self.theta = [] # model's weights
        self.err = 0
        
    def cost_function(self, y_real, y_pred): 
        # cost function for gradient descent algorithm
        return np.sum(abs(y_pred-y_real))/(len(y_real))
    
    def gradient_descent_step(self, learning_rate, dy, m, n, X_tr):
        # one gradient descent step
        s = (np.dot(dy.T, X_tr)).reshape(n, 1)
        dtheta = 2*(learning_rate*s/m).reshape(n, 1)
        return self.theta - dtheta

    
    def fit(self, X, y, learning_rate = 0.01, nsteps = 10000, e = 0.000000001,
            weight_low = 0, weight_high = 1):
        
        np.random.seed(0)
        X = X.astype(float)
        m = X.shape[0]
        # add one's column to X
        X = np.hstack( (np.ones(m).reshape(m, 1), X) )
        n = X.shape[1]
        
        # thetaeights: random initialization
        self.theta = np.random.randint(low = weight_low, high = weight_high, size=(n, 1))
            
        y_pred = np.dot(X, self.theta)
        cost0 = self.cost_function(y, y_pred)
        y = y.reshape(m, 1)
        k = 0
        
        ########## Gradient descent's steps #########
        while True:
            dy = y_pred - y
            theta_tmp = self.theta
            self.theta = self.gradient_descent_step(learning_rate, dy, m, n, X)
            y_pred = np.dot(X, self.theta)
            cost1 = self.cost_function(y, y_pred)
            k += 1
            if (cost1 > cost0):
                self.theta = theta_tmp
                break    
                
            if ((cost0 - cost1) < e) or (k == nsteps):
                break
                
            cost0 = cost1
        #############################################
        return self.theta # return model's weights
    
    def predict(self, X):
        m = X.shape[0]
        return np.dot( np.hstack( (np.ones(m).reshape(m, 1),X.astype(float)) ) ,self.theta)




