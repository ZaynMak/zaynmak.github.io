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
        # X is being transposed to make it a column vector
        # X_ is the augmented matrix
        # The first step is to add a column of 1s to the matrix X
        # The second step is to initialize the weight vector with random values between 1 and 3
        n = X.shape[0]
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        self.weight = np.random.rand(1, 3)
        
        # The third step is to iterate over the data and update the weight vector
        # For each iteration, a random row is selected from the augmented matrix
        # If the dot product of the weight vector and the row is less than 0, the weight vector is updated by adding X_'s value at that row
        # If the dot product of the weight vector and the row is greater than or equal to 0, the weight vector is updated by subtracting X_'s value at that row
        for _ in range(max_steps):
            i = np.random.randint(n)
            if np.dot(self.weight, X_[i].T) < 0:
                self.weight += X_[i]
            else:
                self.weight -= X_[i]
            score = self.score(X_, y)
            self.history.append(score)

            # at the end of every iteration, the score is checked
            # if the score is 1, the loop is broken
            if score == 1:
                break