import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from inverseClassifier import inverseClassifier


center_box = (2, 2)
X,y = datasets.make_blobs(n_samples=50, n_features=2, centers= [(-1,-1), (2,2)], cluster_std=1.05, random_state=40)
# print(X)
y = np.where(y == 0, -1, 1)

clf = inverseClassifier()
plot_data = clf.fit(X, y)

print(clf.alpha, clf.c)

def visualize():

    fig, graph = plt.subplots(2, 2)


    # plot data points

    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    # plt.scatter(X[:, 0], X[:, 1], c=y)

    # plot Classification Curve
    # clf.c = .5
    # clf.alpha = 1

    ## Ploting Decision Boundry and Data set
    graph[0, 0].set_title('Data Set and Decision Boundry')

    if clf.c > x0_1 and clf.c < x0_2:
        x1 = np.linspace(1.1*clf.c, x0_2 , 100)
        x2 = np.linspace(x0_1, 0.9*clf.c, 100)
        y1_ = clf.alpha*(1/(x1-clf.c)+ 1/(clf.c-1))
        y2_ = clf.alpha*(1/(x2-clf.c)+ 1/(clf.c-1))
        graph[0, 0].plot(x1, y1_)
        graph[0, 0].plot(x2, y2_)
    else:
        x = np.linspace(x0_1, x0_2 , 100)
        y_ = clf.alpha*(1/(x-clf.c)+ 1/(clf.c-1))
        graph[0,0].plot(x, y_)
    
    graph[0, 0].scatter(X[:, 0], X[:, 1], c=y)

    ## Plotting Loss Over Iteration

    graph[1, 0].plot(plot_data[:,0], plot_data[:, 1])
    graph[1, 0].set_title('MSE vs iterations')

    ##Plotting Accuracy Over Iteration

    graph[1, 1].plot(plot_data[:,0], plot_data[:, 2])
    graph[1, 1].set_title('Accuracy vs iterations')

    ## Plotting ROC Curve
    
    graph[0, 1].plot(plot_data[:,5], plot_data[:, 6])
    graph[0, 1].set_title('ROC Curve')

    plt.show()

visualize()