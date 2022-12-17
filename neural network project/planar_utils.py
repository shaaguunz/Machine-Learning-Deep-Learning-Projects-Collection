import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

def plotDecisionBoundary(X, y, xaxislabel, yaxislabel, plotdecboundary=False, model= None, debug=0):

    if debug:
      return
    # Set min and max values and give it some padding
    K = len(np.unique(y))

    font = {'family': 'Arial',
            'size': 22}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(12, 10))

    X = X.T
    y = y.reshape(1, y.shape[0]).ravel()

    cols = ['blue', 'crimson', 'green', 'orange', 'magenta','dimgray','goldenrod']
    bkcols = ['lightskyblue', 'lightcoral', 'palegreen', 'papayawhip', 'plum','gainsboro','wheat']


    if plotdecboundary:

      ff = 0.2
      x_min, x_max = X[0, :].min() - ff, X[0, :].max() + ff
      y_min, y_max = X[1, :].min() - ff, X[1, :].max() + ff
      h = 0.02
      # Generate a grid of points with distance h between them
      xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
      # Predict the function value for the whole grid
      dd = np.c_[xx.ravel(), yy.ravel()]

      Z = model(dd).reshape(xx.shape)


      plt.contourf(xx, yy, Z, colors=bkcols, levels=[-0.5] + [k for k in range(K)])#cmap=plt.cm.Spectral)


    for k in range(K):

        Xpos = X.T[y.ravel() == k]

        X1pos = Xpos[:,0:1]
        X2pos = Xpos[:,1:2]

        x1posplot = np.array(X1pos).ravel()
        x2posplot = np.array(X2pos).ravel()


        plt.scatter(x1posplot, x2posplot, edgecolors=cols[k], linewidth=5, s=90, marker='o', facecolors="none", label="$y={}$".format(k))


    plt.xlabel(xaxislabel)
    plt.ylabel(yaxislabel)
    plt.legend(loc=0)
    plt.show()
        

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    #X = X.T
    #Y = Y.T

    return X, Y

def load_multiclass_dataset(N=200, K=2):
  gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=K, shuffle=True, random_state=None)
  return gaussian_quantiles

def load_extra_datasets(N=200):
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    #no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles#, no_structure