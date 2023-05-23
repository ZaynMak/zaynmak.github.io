import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.score_history = []

    def fit_analytic(self, X, y):
        X = pad(X)
        self.w = np.random.rand(1, X.shape[1])[0]
        self.w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    def fit_gradient(self, X, y, batch_size, alpha = 0.01, max_iter = 100):
        n = X.shape[0]
        X_ = pad(X)
        self.w = np.random.rand(1, X_.shape[1])[0]

        # we iterate over the number of epochs and perform the gradient step
        for j in np.arange(max_iter):
            # we shuffle the order of the examples
            order = np.arange(n)
            np.random.shuffle(order)

            # we iterate over the batches and perform the gradient step
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]

                P = np.dot(x_batch.T, x_batch)
                q = np.dot(x_batch.T, y_batch)
                grad = self.gradient(P, q)

                # perform the gradient step
                self.w -= alpha * grad
                if len(self.w) != X_.shape[1]:
                    raise Exception(f"Number of initial weights passed does not match number of features: {len(self.w)}!={X_.shape[1]}")

            # we keep track of the score
            self.score_history.append(self.score(X, y))
    
    def gradient(self, P, q):
        return 2 * (np.dot(P, self.w) - q)

    def predict(self, X):
        return np.dot(self.w, X.T)

    def score(self, X, y):
        X = pad(X)
        y_pred = self.predict(X)
        num = np.linalg.norm(y_pred - y) ** 2
        y_mean = sum(y) / len(y)
        den = np.linalg.norm(y_mean - y) ** 2
        return 1 - (num / den)    

def pad(X):
    return np.append(X, np.ones((X.shape[0], 1)), 1)