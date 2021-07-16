from scipy import spatial
from numpy.random import randn,randint #importing randn

import numpy as np #importing numpy
import matplotlib.pyplot as plt #importing plotting module
import itertools
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import kde
from sklearn.neighbors import KNeighborsClassifier
import timeit
import statistics

def plotDensity_2d(X,Y):
    nbins = 200
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    xi, yi = np.mgrid[minx:maxx:nbins*1j, miny:maxy:nbins*1j]
    def calcDensity(xx):
        k = kde.gaussian_kde(xx.T)        
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        return zi.reshape(xi.shape)
    pz=calcDensity(X[Y==1,:])
    nz=calcDensity(X[Y==-1,:])
    
    c1=plt.contour(xi, yi, pz,cmap=plt.cm.Greys_r,levels=np.percentile(pz,[75,90,95,97,99])); plt.clabel(c1, inline=1)
    c2=plt.contour(xi, yi, nz,cmap=plt.cm.Purples_r,levels=np.percentile(nz,[75,90,95,97,99])); plt.clabel(c2, inline=1)
    plt.pcolormesh(xi, yi, 1-pz*nz,cmap=plt.cm.Blues,vmax=1,vmin=0.99);plt.colorbar()
    markers = ('s','o')
    plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
    plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')   
    #
    plt.grid()
    plt.show()
                   

def plotit(X,Y=None,clf=None, markers = ('s','o'), hold = False, transform = None):
    """
    Just a function for showing a data scatter plot and classification boundary
    of a classifier clf
    """
    eps=1e-6
    minx, maxx = np.min(X[:,0]), np.max(X[:,0])
    miny, maxy = np.min(X[:,1]), np.max(X[:,1])
    
    if clf is not None:
        npts = 150
        x = np.linspace(minx,maxx,npts)
        y = np.linspace(miny,maxy,npts)
        t = np.array(list(itertools.product(x,y)))
        if transform is not None:
            t = transform(t)
        z = clf(t)
        z = np.reshape(z,(npts,npts)).T        
        extent = [minx,maxx,miny,maxy]
        plt.contour(x,y,z,[-1+eps,0,1-eps],linewidths = [2],colors=('b','k','r'),extent=extent, label='f(x)=0')
        #plt.imshow(np.flipud(z), extent = extent, cmap=plt.cm.Purples, vmin = -2, vmax = +2); plt.colorbar()
        plt.pcolormesh(x, y, z,cmap=plt.cm.Purples,vmin=-2,vmax=+2);plt.colorbar()
        plt.axis([minx,maxx,miny,maxy])
    
    if Y is not None:
        
        plt.scatter(X[Y==1,0],X[Y==1,1],marker = markers[0], c = 'y', s = 30)
        plt.scatter(X[Y==-1,0],X[Y==-1,1],marker = markers[1],c = 'c', s = 30)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')        
         
    else:
        plt.scatter(X[:,0],X[:,1],marker = '.', c = 'k', s = 5)
    if not hold:
        plt.grid()
        
        plt.show()
    
def accuracy(ytarget,ypredicted):
    return np.sum(ytarget == ypredicted)/len(ytarget)


class NN:
    def __init__(self):
        pass
    def fit(self, X, Y):
        self.Xtr=X
        self.Ytr=Y
        
    def predict(self, Xts, k):
        Yts=[]
        Xts=np.array(Xts)

        if(k == 1):
            for t in Xts:
                distances=np.sqrt(np.sum(np.power((self.Xtr - t), 2), axis=1)) #euclidean distance
                y=self.Ytr[np.argmin(distances)] #index of minimum distance
                Yts.append(y)
            return Yts

        else:
            data=[]
            for t in Xts:
                dist= np.sqrt(np.sum(np.power((self.Xtr - t), 2), axis=1))
                m=dist[np.argmax(dist)]
                #print("Main Distance:",dist)
                #print("Main YTR",self.Ytr)
                min= np.argmin(dist)
                #print("min=>",dist[min])
                s= self.Ytr[min]
                data.append(s)


                for i in range(1, k):
                    # dist = np.delete(dist, min)
                    # self.Ytr = np.delete(self.Ytr, min)
                    dist[min]=m
                    # print("Distance:", dist)
                    # print("YTR", self.Ytr)
                    min= np.argmin(dist)
                    #print("min=>",dist[min])
                    s2= self.Ytr[min]
                    data.append(s2)


                positive = negative = 0

                for i in data:
                    if(i == 1):
                        positive += 1
                    else:
                        negative += 1

                if(positive > negative):
                    Yts.append(1)
                else:
                    Yts.append(-1)
            return Yts


def getExamples(n=100,d=2):
    """
    Generates n d-dimensional normally distributed examples of each class        
    The mean of the positive class is [1] and for the negative class it is [-1]
    DO NOT CHANGE THIS FUNCTION
    """
    Xp = randn(n,d)+1   #generate n examples of the positie class
    #print("XP", Xp)
    #Xp[:,0]=Xp[:,0]+1
    Xn = randn(n,d)-1   #generate n examples of the negative class
    #print("XN", Xn)
    #Xn[:,0]=Xn[:,0]-1
    X = np.vstack((Xp,Xn))  #Stack the examples together to a single matrix
    #print("X",X)
    Y = np.array([+1]*n+[-1]*n) #Associate Labels
    #print("Y",Y)
    return (X,Y) 


def Q2_A(n, k, d):
    times=[]
    sk_times=[]
    samples=[]

    for i in range(2,300):
        Xtr, Ytr = getExamples(n=i, d=d)
        Xtt, Ytt = getExamples(n=n, d=d)

        start = timeit.default_timer()
        clf = NN()
        clf.fit(Xtr, Ytr)
        predict_Ytt = clf.predict(Xtt, k)
        E = accuracy(Ytt, predict_Ytt)
        end = timeit.default_timer()
        total = end - start
        samples.append(i)

        start_SKL = timeit.default_timer()
        skl = KNeighborsClassifier(n_neighbors=k)
        skl.fit(Xtr, Ytr)
        Y = skl.predict(Xtt)
        C = accuracy(Ytt, Y)
        end_SKL = timeit.default_timer()
        total_SKL = end_SKL - start_SKL

        times.append(round(float(total),2))
        sk_times.append(round(float(total_SKL),2))

    plt.plot(samples,times, label="Simple")
    plt.plot(samples,sk_times, label="SkLearn")
    plt.legend(loc='upper left')
    plt.title("increase in number of training examples", fontsize=14, fontweight='bold')
    plt.xlabel("No. of Samples")
    plt.ylabel("Execution time in seconds")
    plt.show()


def Q2_B(n, k):
    times=[]
    sk_times=[]
    samples=[]

    for i in range(2,100):
        Xtr, Ytr = getExamples(n=n, d=i)
        Xtt, Ytt = getExamples(n=n, d=i)

        start = timeit.default_timer()
        clf = NN()
        clf.fit(Xtr, Ytr)
        predict_Ytt = clf.predict(Xtt, k)
        E = accuracy(Ytt, predict_Ytt)
        end = timeit.default_timer()
        total = end - start
        samples.append(i)

        start_SKL = timeit.default_timer()
        skl = KNeighborsClassifier(n_neighbors=k)
        skl.fit(Xtr, Ytr)
        Y = skl.predict(Xtt)
        C = accuracy(Ytt, Y)
        end_SKL = timeit.default_timer()
        total_SKL = end_SKL - start_SKL

        times.append(round(float(total),2))
        sk_times.append(round(float(total_SKL),2))

    plt.plot(samples,times, label="Simple")
    plt.plot(samples,sk_times, label="SkLearn")
    plt.legend(loc='upper left')
    plt.title("increase in dimensionality", fontsize=14, fontweight='bold')
    plt.xlabel("No. of Samples")
    plt.ylabel("Execution time in seconds")
    plt.show()


def Q2_C(n, d):
    times=[]
    sk_times=[]
    Xtr, Ytr = getExamples(n=n, d=d)
    clf = NN()

    for i in range(1, 100, 2):

        clf.fit(Xtr, Ytr)
        predict_Ytt = clf.predict(Xtr, i)
        E = accuracy(Ytr, predict_Ytt)

        skl = KNeighborsClassifier(n_neighbors=i)
        skl.fit(Xtr, Ytr)
        Y = skl.predict(Xtr)
        C = accuracy(Ytr, Y)

        times.append(E)
        sk_times.append(C)

    k=np.array(range(1,100,2))
    plt.title("training accuracy due to change in K", fontsize=14, fontweight='bold')
    plt.plot(k, times, label="Simple")
    plt.plot(k, sk_times, label="SkLearn")
    plt.legend(loc='center right')
    plt.xlabel("No. of k")
    plt.ylabel("Accuracy")
    plt.show()


def Q2_D(n, d):
    times=[]
    sk_times=[]
    Xtr, Ytr = getExamples(n=n, d=d)
    Xtt, Ytt = getExamples(n=n, d=d)

    for i in range(1, 100, 2):
        clf = NN()
        clf.fit(Xtr, Ytr)
        predict_Ytt = clf.predict(Xtt, i)
        E = accuracy(Ytt, predict_Ytt)

        skl = KNeighborsClassifier(n_neighbors=i)
        skl.fit(Xtr, Ytr)
        Y = skl.predict(Xtt)
        C = accuracy(Ytt, Y)

        times.append(E)
        sk_times.append(C)

    k=np.array(range(1,100,2))
    plt.title("testing accuracy due to change in K", fontsize=14, fontweight='bold')
    plt.plot(k, times, label="Simple")
    plt.plot(k, sk_times, label="SKLearn")
    plt.legend()
    plt.xlabel("No. of k")
    plt.ylabel("Accuracy")
    plt.show(loc='center right')




if __name__ == '__main__':
    #%% Data Generation and Density Plotting
    g = input("Enter value of 'k' : ")
    k = int(g)

    n = 100 #number of examples of each class
    d = 2 #number of dimensions
    Xtr,Ytr = getExamples(n=n,d=d) #Generate Training Examples    
    print("Number of positive examples in training: ", np.sum(Ytr==1))
    print("Number of negative examples in training: ", np.sum(Ytr==-1))
    print("Dimensions of the data: ", Xtr.shape[1])
    Xtt,Ytt = getExamples(n=n,d=d) #Generate Testing Examples
    plt.figure()
    plotDensity_2d(Xtr,Ytr)
    plt.title("Train Data")
    
    plt.figure()
    plotDensity_2d(Xtt,Ytt)
    plt.title("Test Data")
    #%% Nearest Neighbour
    #Classify    
    #print("*"*10+"1- Nearest Neighbor Implementation"+"*"*10)
    clf = NN()
    clf.fit(Xtr,Ytr)
    predict_Ytt = clf.predict(Xtr, k)

    #print("*"*10+"1- SKLearn Implementation"+"*"*10)
    skl = KNeighborsClassifier(n_neighbors=k)
    skl.fit(Xtr,Ytr)
    Y = skl.predict(Xtt)

    print("Actual:", Ytt)
    print("Predict:", predict_Ytt)
    #Evaluate Classification Error
    E = accuracy(Ytr,predict_Ytt)
    print("trainAccuracy", E)

    tE = accuracy(Ytt, predict_Ytt)
    print("testAccuracy", tE)
    C = accuracy(Ytt,Y)
    print("SKLearn Accuracy", C)

    #Q2_A(n, k, d)
    Q2_B(n, k)
    # Q2_C(n, d)
    # Q2_D(n, d)

    voronoi_plot_2d(Voronoi(Xtr),show_vertices=False,show_points=False,line_colors='orange')
    # plotit(Xtr,Ytr,clf=clf.predict)
    # plt.title("1-NN  Implementation Train Data")
    # plt.figure()
    # plotit(Xtt,Ytt,clf=clf.predict)
    # plt.title("1-NN  Implementation Test data")


