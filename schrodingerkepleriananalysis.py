import itertools
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab as plt
import sklearn
import scipy as sp
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy.sparse import csgraph
#from pyemd import emd
from scipy import stats
from scipy.stats import norm
from sklearn import cluster
from numpy import linalg as la 

testdata= [ [0,87.9691,0,0,0.00017,0], [0,224.70069,0,0,0.00256,0], [0,365.25636,0,0,0.00315,0], [0,686.971,0,0,0.00034,0],
            [0,4331.572,0,0,1,0], [0,10759.22,0,0,0.299,0], [0,30799.095,0,0,0.046,0], [0,60190.0,0,0,0.054,0] ,[0,90613.305,0,0,0.00000686512,0]]

##################################################################################
#                                                                                #
#                                  OVERVIEW                                      #
#                                                                                #
#               The code requires knownplanets.csv. The file is parsed           #
#            as a list of lists where each list is the data pertaining to        #
#            one planet. Rank values are calculated and collected followed       #
#            by generating integer probability distribution and uniform          #
#            distribution from 0 to 0.5. The KL divergence is measured and       #
#            relivent histograms are plotted. Following this the eccentricties   #
#            of all planets are collected and the theoreticaly predicted         #
#            ecentricities calculated. The KL divergence is then taken between   #
#            these two sets. An adjacency matrix is then constructed whose       #
#            edges are as defined in the report. In order to change edge         #
#            conditions you must manually do it in the if condistionals in       #
#            the adjacencymatrix() function. Necessary calculations are made     #
#            to find the normalised Laplacian and eigenvector matrix. From       #
#            this the graphs are plotted.                                        #
#                                                                                #
##################################################################################


#
# Parse the .CSV file into a list of lists. Each list is the info of one planet
# planets[0] = ['NAME', 'PER', 'ECC', 'OM', 'MASS', 'MSTAR', 'BINARY', 'BINARYURL', 'BINARYREF']
# planets[1] = ['', 'day', '', 'deg', 'mjupiter', 'msun', '', '', '']
#
knownplanets = []
with open('./data/knownplanets.csv', 'rb') as csvfile:
    planet =[]
    planetparser = csv.reader(csvfile)
    for line in planetparser:
        newline = []
        for x in line:
            try:
                x = float(x)
                newline.append(x)
            except ValueError:
                x = x
                newline.append(x)
        knownplanets.append(newline)
fields = knownplanets.pop(0)
units = knownplanets.pop(0)


def collecteccentricities():
    eccentricities = []
    for planet in knownplanets:
        eccentricities.append(planet[2])
    return(eccentricities)

unconfirmed = []   
with open('./data/unconfirmedplanets.csv', 'rb') as csvfile:
    planet =[]
    planetparser = csv.reader(csvfile)
    for line in planetparser:
        newline = []
        for x in line:
            try:
                x = float(x)
                newline.append(x)
            except ValueError:
                x = x
                newline.append(x)
        unconfirmed.append(newline)


#
# Calculate n = n_0 = min[sum_{0}^{planets in solar system] (n_i gamma - int( n_i gamma )  )^2]
#...
#The value of n_0 is varied from 0 to 10 in search for a minimum of the sum of squared differences
# between the calculated rank n and the closest integer.
def minsumsquare(planets, base):
    for ni in range(11):
        minsumsquare = float("inf")
        ntemp = 0
        for planet in planets:
            if(False == isinstance(planet[4], str)):
                n_temp = n(float(ni), float(planet[1]), float(planet[4]), base)
                dif = n_temp - int(n_temp)
                ntemp += np.power(dif, 2)
        if(ntemp <= minsumsquare):
            minsumsquare = ntemp
    return(minsumsquare)


#
# Calculate n = n_0 [ (t/t_o) / (m/m_0) ]^(1/3)
#,4331.572,0,0,1
#
def n(n_o, t, m, base):
    if(base == 'earth'):
        t_o = float(365.25636)
        m_o = float(0.00315)
    else:
        t_o = float(4331.572) 
        m_o = 1
    a = (t/t_o)#/(float(m)/m_o)
    b = 1/float(3)
    return(n_o*np.power(a,b))

#
# Calculate the rank for all planets in data set
#
def nall( planets , base ):
    n_o = minsumsquare(testdata,base)
    nlist = []
    nactual = []
    for planet in planets:
        n_ = n(n_o, planet[1], planet[4], base)
        nactual.append(n_)
        nlist.append(round(n_))
    return([nactual, nlist])

#
# Calculate the difference between the nearest integer rank and true value of calculated rank
#
def variance(ntrue, nlist):
    differencelist = []
    i = 0
    for n in ntrue:
        diff = np.abs(n - nlist[i])
        differencelist.append(diff)
        i += 1
    return(differencelist)

#
# Use the Earth Movers metric between a uniform distribution between 0 and 1
# and the distance of calculated ranks to 1
#
def earthmoverscluster(planets):
    nlist = nall(knownplanets)[1]
    ntrue = nall(knownplanets)[0]
    difference = variance(ntrue,nlist)
    uniformdist = np.asarray(np.random.uniform(0.0,1.0,len(difference)))
    difference = np.asarray(difference)
    pointsdifference = []
    pointsuniform = []
    for i in range(len(difference)):
        pointsdifference.append([0 , difference[i]])
        pointsuniform.append([0,uniformdist[i]])
    #print(emd(pointsdifference,pointsuniform))
    return()

#
# Compute the relative entropy (kullback-Leibler divergence) between the distance from integer values
# of calculated ranks and a uniform distribution between 0 and 0.5
#
def kldistancecluster(planets):
    nlist = nall(knownplanets, 'earth')[1]
    ntrue = nall(knownplanets, 'earth')[0]
    difference = variance(ntrue,nlist)
    uniformdist = np.asarray(np.random.uniform(0.0,0.5,len(difference)))
    difference = np.asarray(difference)

    plt.hist(nlist, bins = 25, color = 'blue', alpha = 0.7, normed = True)
    plt.hist(ntrue, bins = 25, color = 'green', alpha = 0.5, normed = True)
    
    # Find best fit
    x = np.linspace(0.0, 8, 25)
    best_fit_uniform = mlab.normpdf(x, np.mean(nlist), np.std(nlist))
    best_fit_dif = mlab.normpdf(x, np.mean(ntrue), np.std(ntrue))
    plt.plot(x, best_fit_uniform, label = 'unif')
    plt.plot(x, best_fit_dif, label = 'dif')
    plt.xlabel('Distribution Value')
    plt.ylabel('Frequency')
    
    blue = mpatches.Patch(color='blue', label = 'Normed PDF for Integer Distribution')
    green = mpatches.Patch(color = 'green', label = 'Normed PDF for Calculate Rank')
    plt.legend(handles = [blue, green])
   
    plt.text(3.4, 1.0, 'KL Divergence: \n  ( rank, integer distribution) = 0.0112', style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.show()
    #kldiv = stats.entropy(difference, qk=uniformdist, base=None)
    kldiv = stats.entropy(nlist, qk=ntrue, base=None)
    return(kldiv)


#
# Gather the eccentricities (e) of each planet and calculate the rank (n) of 
# each planet. Test if the eccentricity abides by Scale Relativities prediction 
# that e = k/n where k is an integer ranging from 0 to n-1. 
#
def eccentricityrungelenz():
    nlist = nall(knownplanets,'earth')[1]
    ntrue = nall(knownplanets,'earth')[0]
    eccentricities = collecteccentricities()
    
    runglenz = []
    predictedeccentricity = []
    predictedeccentricityintegern = []
    index = 0
    for e in eccentricities:
        if False == isinstance(e, str):
            predeintn = [e]
            prede = [e]                      # form a list of all possible 
            for k in range(int(nlist[index])): # predicted eccentricies with the
                prede.append(k/ntrue[index]) # true eccentricity at index=0
                predeintn.append(k/nlist[index])
            predictedeccentricity.append(prede)
            predictedeccentricityintegern.append(predeintn)
        index += 1
    # identify the closest predicted eccentricity to the actual observed value
    nearestpredictede = []
    actualeccentricity = []
    difobservedpredicted = []
    for elist in predictedeccentricity:
        mindif = 10.0
        actuale = elist[0]
        elist = elist[1:]
        for i in range(len(elist)):
            e = float(elist[i])
            dif = abs(actuale - elist[i])
            if dif < mindif:
                mindif = dif
                neareste = e
        actualeccentricity.append(actuale)
        nearestpredictede.append(neareste)
        difobservedpredicted.append(mindif)
        
    return([actualeccentricity, nearestpredictede, difobservedpredicted,
            predictedeccentricity])

def kldistanceeccentricity():       
    eccentricityinfo = eccentricityrungelenz()
    actualeccentricity = np.asarray(eccentricityinfo[0])
    nearestpredicted = np.asarray(eccentricityinfo[1])
    difobservedpredicted = np.asarray(eccentricityinfo[2])

    
    uniformdist = np.asarray(np.random.uniform(0.0,1.0,len(nearestpredicted)))
    miscore = sklearn.metrics.mutual_info_score(actualeccentricity, nearestpredicted)
    # find the kl divergence between the observed eccentricities and theoretical predictions
    kldiv = stats.entropy(actualeccentricity, qk=nearestpredicted, base=None)
    
    return(kldiv)

nlist = nall(knownplanets, 'earth')[1]
ntrue = nall(knownplanets, 'earth')[0]
var = variance(ntrue, nlist)

#print(kldistanceeccentricity())

def nhist():
    plt.hist( nlist, 25, normed = False, alpha = 0.7)
    plt.title("Calculated Values of Rank")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.show()

def ndifhist():
    plt.hist(var, 75, normed = False, alpha = 0.5)
    plt.title("Deviation of Ranks from Nearest Integer")
    plt.xlabel("Diference n-int(n)")
    plt.ylabel("Frequency")
    plt.show()


#plt.scatter(nlist,range(len(nlist)))
#plt.hist(var, 10, normed = True, alpha = 0.5)
#plt.show()

#plt.plot(range(11), x, linewidth = 2.0)
#plt.show()

#print(kldistancecluster(knownplanets))
def eccentricityinfoplot():
    eccentricityinfo = eccentricityrungelenz()
    actualeccentricity = np.asarray(eccentricityinfo[0])
    nearestpredicted = np.asarray(eccentricityinfo[1])
    difobservedpredicted = np.asarray(eccentricityinfo[2])

    #plt.figure(1)
    plt.hist(actualeccentricity, 40, normed = False, color = "blue",
             label = "Observed Eccentricities from Exoplanet Database")# range(len(actualeccentricity)))
    (mu, sigma) = norm.fit(actualeccentricity)
   # y = mlab.normpdf(40,mu,sigma)
    #plt.plot(40,y,'r--',linewidth=2)
    
    plt.hist(nearestpredicted, 40, normed = False, color = "green", alpha = 0.7,
             label = "Scale Relativities Predicted Eccentricities")

    plt.title("Observed Ecentricities and Theoretical Predicted Eccentricities")
    plt.ylabel("Frequency of Observed Eccentricity")
    plt.xlabel("Eccentricity")
    plt.legend()
    plt.show()
    
#eccentricityinfoplot()

def nearthsystemplot():
    earthbase = nall(testdata, 'earth')
    jupiterbase = nall(testdata, 'jupiter')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter( range(len(earthbase[0])),earthbase[0], s=10, c='b', marker="s", label='Earth as Reference')
    ax1.scatter(range(len(jupiterbase[1])),jupiterbase[0], s=10, c='r', marker="o", label='Jupiter as Reference')
    plt.xlabel("Planet Number")
    plt.ylabel("Rank Value")
    plt.title("Dependence on Reference Planet")
    plt.legend(loc='upper left');
    plt.show()

################################################################################
#
#                   We know define a graph for the planets within                   #
#                   the exoplanet database whose nodes are each planet              #
#                   and two nodes form an edge if
#                       (i) The masses are within some epsilon value of eachother   #
#                       (ii) The periods are within some delta value of eachother   #
#                       (iii) The mass of the two suns are within some gamma value  #
#                   We then will define an adjacency matrix based on the previous   #
#                   three conditions. From the adjacency matrix we can then cluster #
#                   the planets based on the eigenvectors of the adjacency matrix's #
#                   Laplacian matrix. 
#
#####################################################################################

#
# The adjacency matrix is constructed based on the condistion in the if condistions.
# all condistions are centered around planetary data values being within some epsilon
# or delta of one another. The code is written in such a way that any number of data
# elements from the exoplanet database can be used irregardles of units. 
# Returns: list(adjacencymatrix, degreematrix)
#
def adjacencymatrix():
    #
    # Create a matrix of all zeros which is number of planets X number of planets
    #
    adjacencymatrix = np.array(np.zeros(len(knownplanets)-1))
    degreematrix = np.array(np.zeros(len(knownplanets)-1))
    for i in range(len(knownplanets)-2):
        adjacencymatrix = np.vstack((adjacencymatrix, np.zeros(len(knownplanets)-1)))
        degreematrix = np.vstack((degreematrix, np.zeros(len(knownplanets)-1)))

    epsilon = 1/float(100)
    delta = float(5)

    row = 0
    for planet in knownplanets:
        mass = planet[4]
        per = planet[1]
        mstar = planet[5]
        col = 0
        rowcheck = row < len(adjacencymatrix)
        for planet in knownplanets:
            colcheck = col<len(adjacencymatrix)
            if (isinstance(planet[4],float)) and (isinstance(mass, float)) and colcheck and rowcheck:
                if (abs(planet[1] - per) < delta) and (abs(planet[4] - mass) < epsilon):
                    adjacencymatrix[row,col] = float(1)
            if row == col and colcheck and rowcheck:
                adjacencymatrix[row,col] = float(0)
            col = col + 1
        row = row + 1
        
    for i in range(len(adjacencymatrix)):
            degreematrix[i,i] = np.sum(adjacencymatrix[i])


    return([adjacencymatrix, degreematrix])

#
# The function creates an adjacency matrix and degree matrix followed by
# calculating the normed Laplacian L = I - D^{-1/2}AD^{-1/2}. Following
# this the eigenvalues and eigenvectors are computed.
# Returns: list( np.array(eigenvalues), np.array(eigenvectors) )
#
def adjacencylaplacian():
    adjacencyresults = adjacencymatrix()
    adjacencym = adjacencyresults[0]
    degreematrix = adjacencyresults[1]
    Dnegroot = np.identity(len(degreematrix))
    for i in range(len(degreematrix)):
        if degreematrix[i,i] != 0:
            Dnegroot[i,i] = 1/math.sqrt(float(degreematrix[i,i]))

    I = np.identity(len(adjacencym))

    #L = np.dot(np.dot(Dnegroot,adjacencym),Dnegroot)
    laplacian = I - np.dot(Dnegroot,np.dot(adjacencym,Dnegroot))
    #laplacian = csgraph.laplacian(L, normed = True)
    eigenvals, eigenvec = la.eig(laplacian)
    
    return([eigenvals,eigenvec])

#
# The function computes the adjacency matrix and its accompanying
# eigenvalues and vectors. The second and third eigenvector define
# the position of the nodes defined by the adjacency matrix.
# Returns: Spectral Cluster Graph
#
def plotadjacencymatrix( ):
    #adjacencygraph = nx.from_scipy_sparse_matrix(adjacencymatrix)
    #adjacency_matrix = nx.to_numpy_matrix(adjacencymatrix()[0], dtype = np.bool)
    adjacency_matrix = adjacencymatrix()[0]
    eigeninfo = adjacencylaplacian()
    eigenval = list(eigeninfo[0])

    eigenvec = eigeninfo[1]

    copyeigenval = sorted(eigenval)
    for i in range(len(copyeigenval)):
        copyeigenval[i] = float(copyeigenval[i])
        eigenval[i] = float(eigenval[i])
        
    c0 = list(eigenval).index(copyeigenval[0])
    c1 = list(eigenval).index(copyeigenval[1])
    c2 = list(eigenval).index(copyeigenval[2])

    #lambdaone = 1/float(math.sqrt(abs(eigenval[c1])))
    #lambdatwo = 1/float(math.sqrt(abs(eigenval[c2])))
    lambdaone = 1/float(math.sqrt(abs(eigenval[1])))
    lambdatwo = 1/float(math.sqrt(abs(eigenval[2])))
    
    #col12 = [[float(lambdaone*eigenvec[0,c1]),float(lambdatwo*eigenvec[1,c2])]]
    col12 = [[float(lambdaone*eigenvec[0,1]),float(lambdatwo*eigenvec[1,2])]]
    #print(col12)

    for row in eigenvec[1:]:
        #point = [float(lambdaone*row[c1]) , float(lambdatwo*row[c2])]
        point = [float(lambdaone*row[1]) , float(lambdatwo*row[2])]
        col12.append(point)

    pos = col12
    G = nx.DiGraph(adjacency_matrix)
    
    nx.draw(G,pos)
    plt.show()

#################################################################################
#                                                                               #
#                     Function calls for data retreival                         #
#                                                                               #
#################################################################################
#plotadjacencymatrix()
#adjacencylaplacian()
#A = adjacencymatrix()[0]
kldistancecluster(knownplanets)
