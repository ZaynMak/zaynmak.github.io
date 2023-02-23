import numpy as np

class Perceptron:

    def __init__(self):
        self.weight = None
        self.history = []
        
    def predict(self, X):
        #  X is bring transposed to make it a column vector
        return (np.dot(self.weight, X.T) >= 0).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def fit(self, X, y, max_steps = 1000):
        n = X.shape[0]
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.weight = np.random.rand(1, 3)
        
        for _ in range(max_steps):
            i = np.random.randint(n)
            if np.dot(self.weight, X_[i].T) < 0:
                self.weight += X_[i]
            else:
                self.weight[0] -= X_[i]
            score = self.score(X_, y)
            self.history.append(score)
            if score == 1:
                break