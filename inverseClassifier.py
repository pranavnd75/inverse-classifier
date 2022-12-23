import numpy as np

class inverseClassifier:
    def __init__(self, learning_rate=0.001, n_iter=1000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.alpha = None
        self.c = None
        # self.a =None
        
    
    def fit(self, X, y):
        self.alpha = 1
        self.c = 0.5
        # self.a = 2
        self.plot_data = np.zeros((self.n_iter, 7))
        for i in range(self.n_iter):
            for idx, x_i in enumerate(X):
                # if x_i[0] > self.c: 
                f_xi = x_i[1] - self.alpha*(1/(x_i[0] - self.c)+ 1/(self.c - 1))
                d_alpha = -2 * (y[idx]-f_xi)*(1/(x_i[0] - self.c)+ 1/(self.c - 1))
                d_c = -2 * self.alpha * (y[idx]-f_xi) * (1/((x_i[0] - self.c)*(x_i[0] - self.c)) - 1/((self.c - 1)*(self.c - 1)))
                self.alpha -= self.lr * d_alpha
                self.c -= self.lr * d_c

                # else:
                #     f_xi = x_i[1] - self.alpha*(1/(x_i[0] - self.c)+ 1/(self.c - 1))
                #     d_alpha = -2*(y[idx]-f_xi)*(1/(x_i[0] - self.c)+ 1/(self.c - 1))
                #     d_c = -2 * self.alpha * (y[idx]-f_xi) * (1/((x_i[0] - self.c)*(x_i[0] - self.c)) - 1/((self.c - 1)*(self.c - 1)))
                #     self.alpha -= self.lr * d_alpha
                #     self.c -= self.lr * d_c
                # When using Hinge Loss Function
                # if f_xi * y_[idx] >= 1:
                    # d_alpha = (y_[idx]) *(1/(x_i[0] - self.c)+ 1/(self.c - 1))
                    # d_c =  (y_[idx]) * self.alpha * (-1/((x_i[0] - self.c)*(x_i[0] - self.c)) + 1/((self.c - 1)*(self.c - 1)))
                # d_alpha = -2*(y_[idx]-f_xi)*(1/(x_i[0] - self.c)+ 1/(self.c - self.a))
                # d_c = -2 * self.alpha * (y_[idx]-f_xi) * (1/((x_i[0] - self.c)*(x_i[0] - self.c)) - 1/((self.c - self.a)*(self.c - self.a)))
                # # d_a = -2 * self.alpha * (y_[idx]-f_xi) * (1/((x_i[0]-self.c)*(x_i[0]-self.c)))
                # self.alpha -= self.lr * d_alpha
                # self.c -= self.lr * d_c
                # self.a -= self.lr * d_a
            
            self.prediction = self.predict(X)
            self.plot_data[i] = np.array([i , self.get_MSE(X, y), self.get_accuracy(X, y), self.get_recall(X,y), self.get_precision(X,y), self.get_true_positive_rate(X,y), self.get_false_positive_rate(X,y)])
            
            print("Running iteration {}, currently: MSE={}, Accuracy={}%, Recall={}, Prescion={}, TruePositiveRate={}, FalseNegativeRate={}".format(i+1, self.plot_data[i][1],self.plot_data[i][2], self.plot_data[i][3], self.plot_data[i][4], self.plot_data[i][6] , self.plot_data[i][5]))
        
        return self.plot_data
    
    def predict(self, X):
        prediction = X[:,1] - self.alpha * (1/(X[:,0]-self.c) + 1/(self.c - 1))
        return np.sign(prediction)
    
    def get_MSE(self, X, y):
        return (1/X.shape[0])* np.sum((y - self.predict(X))*(y - self.predict(X)))*100

    def get_accuracy(self,X, y):
        correct = 0
        for i, row in enumerate(self.prediction):
        # print(prediction[i],y[i])
            if self.prediction[i] == y[i]:
                correct += 1
        return correct/X.shape[0] * 100
    
    def get_precision(self, X, y):
        true_positive = 0
        false_positive = 0
        for i, row in enumerate(self.prediction):
        # print(prediction[i],y[i])
            if y[i] == 1:
                if self.prediction[i] == 1:
                    true_positive += 1
            else:
                if self.prediction[i] == 1:
                    false_positive += 1

        return true_positive/(false_positive + true_positive)
    
    def get_recall(self, X, y):
        true_negative = 0
        false_negative = 0
        
        for i, row in enumerate(self.prediction):
        # print(prediction[i],y[i])
            if y[i] == -1:
                if self.prediction[i] == -1:
                    true_negative += 1
            else:
                if self.prediction[i] == -1:
                    false_negative += 1

        return true_negative/(false_negative + true_negative)
    
    def get_true_positive_rate(self,X,y):
        true_positive = 0
        false_negative = 0
        for i, row in enumerate(self.prediction):
        # print(prediction[i],y[i])
            if y[i] == 1:
                if self.prediction[i] == 1:
                    true_positive += 1
                else:
                    false_negative += 1

        return true_positive/(false_negative + true_positive)
    
    def get_false_positive_rate(self,X,y):
        false_positive = 0
        true_negative = 0
        for i, row in enumerate(self.prediction):
        # print(prediction[i],y[i])
            if y[i] == -1:
                if self.prediction[i] == 1:
                    false_positive += 1
                else:
                    true_negative += 1

        return false_positive/(true_negative + false_positive)



        