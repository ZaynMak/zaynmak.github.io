import numpy as np

class LogisticRegression:

    def __init__(self):
        self.w = None
        self.loss_history = []
        self.score_history = []
        
    def predict(self, X):
        #  X is bring transposed to make it a column vector
        return np.dot(self.w, X.T)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def logistic_loss(self, y_hat, y): 
        # y_hat is the prediction, y is the true value
        # we are calculating the loss for a single example
        return -y * np.log(self.sigmoid(y_hat)) - (1 - y) * np.log(1 - self.sigmoid(y_hat))
    
    def empirical_risk(self, X, y):
        # we are calculating the empirical risk
        y_hat = self.predict(X)
        return np.mean(self.logistic_loss(y_hat, y))

    def score(self, X, y):
        # we are calculating the mean score
        return np.mean((self.predict(X) >= 0).astype(int) == y)

    def gradient(self, X, y):
        # we calculate the gradient
        sigm = (self.sigmoid(self.predict(X)) - y)
        return np.mean(np.multiply(sigm, X.T), axis=1)
    
    def fit(self, X, y, alpha, max_epochs):
        # we add a column of ones to X to account for the bias
        # we initialize the weights randomly
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.random.rand(1, 3)[0]
        
        # we iterate over the number of epochs and perform the gradient step
        # we also keep track of the loss and score
        for i in range(max_epochs):
            self.w -= alpha * self.gradient(X_, y)
            new_loss = self.empirical_risk(X_, y)
            self.loss_history.append(new_loss)
            self.score_history.append(self.score(X_, y))

    
    def fit_stochastic(self, X, y, alpha, batch_size, max_epochs, momentum = False):
        # we add a column of ones to X to account for the bias
        # we initialize the weights randomly
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        n = X.shape[0]
        self.w = np.random.rand(1, 3)[0]
        self.prev_w = self.w.copy()

        # if momentum is true, we set the beta parameter to 0.8 to adjust the weight calculation
        if momentum:
            beta = 0.8
        else:
            beta = 0

        # we iterate over the number of epochs and perform the gradient step
        for j in np.arange(max_epochs):

            # we shuffle the order of the examples
            order = np.arange(n)
            np.random.shuffle(order)


            # we iterate over the batches and perform the gradient step
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                grad = self.gradient(x_batch, y_batch) 
                
                temp = self.w.copy()
                # perform the gradient step
                self.w += ((beta * (self.w + self.prev_w)) - (alpha * grad))
                self.prev_w = temp

            self.loss_history.append(self.empirical_risk(X_, y))

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)