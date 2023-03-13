import numpy as np

class LogisticRegression:

    def __init__(self):
        self.w = None
        self.loss_history = []
        self.score_history = []
        
    def predict(self, X):
        #  X is bring transposed to make it a column vector
        return (np.dot(self.w, X.T) >= 0).astype(int)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def logistic_loss(self, y_hat, y): 
        return -y * np.log(self.sigmoid(y_hat)) - (1 - y) * np.log(1 - self.sigmoid(y_hat))
    
    def empirical_risk(self, X, y):
        y_hat = self.predict(X)
        return np.mean(self.logistic_loss(y_hat, y))

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def gradient(self, X, y):
        sigm = (self.sigmoid(self.predict(X)) - y)
        return np.mean(np.multiply(sigm, X.T), axis=1)
    
    def fit(self, X, y, alpha, max_epochs):
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.w = np.random.rand(1, 3)[0]
        prev_loss = np.inf
        
        for i in range(max_epochs):
            print(self.w)
            print(self.gradient(X_, y))
            self.w -= alpha * self.gradient(X_, y)
            print(self.w)
            new_loss = self.empirical_risk(X_, y)
            self.loss_history.append(new_loss)
            self.score_history.append(self.score(X_, y))

            # if np.isclose(new_loss, prev_loss):
            #     break
            prev_loss = new_loss

        # for _ in range(epochs):
        #     i = np.random.randint(n)
        #     if np.dot(self.w, X_[i].T) < 0:
        #         self.w += X_[i]
        #     else:
        #         self.w[0] -= X_[i]
        #     score = self.score(X_, y)
        #     self.score_history.append(score)
        #     if score == 1:
        #         break

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)