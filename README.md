# prclust  [![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html)  [![CRAN](http://www.r-pkg.org/badges/version/prclust)](http://cran.rstudio.com/package=prclust) [![Downloads](http://cranlogs.r-pkg.org/badges/prclust?color=brightgreen)] (http://www.r-pkg.org/pkg/Rcpp)  ![downloads](http://cranlogs.r-pkg.org/badges/grand-total/prclust)

**prclust** is a new R package that makes it incredibly easy to use penalized regression based clustering method with R. 

## Benefits
* It can treat some complex clustering situations, for example, in the presence of non-convex clusters, in which traditional methods such as K-means break down.
* Apply or modify many established results and techniques, such as model selection criteria, in regression to clustering. 

## Features

* Two algorithms (DC-ADMM and quadratic penalty based algorithm) to implement penalized regression based clustering.
* Two criteria (generalized cross validation and stability based criterion) to select the tuning parameters.
* A function to calculate Rand, aRand and Jaccard index, measuring the agreement between estimated cluster and the truth with a higher value indicating a higher agreement.

## Installation
To install the stable version from CRAN, simply run the following from an R console:

```r
install.packages("prclust")
```

To install the latest development builds directly from GitHub, run this instead:

```r
if (!require("devtools"))
  install.packages("devtools")
devtools::install_github("ChongWu-Biostat/prclust")
```

## Using prclust
Clustering analysis is regarded as unsupervised learning in absence of a class label, as opposed to supervised learning. Over the last few years, a new framework of clustering analysis was introduced by treating it as a penalized regression problem based on over-parameterization. Specifically, we parameterize p-dimensional observations with its own centroid. Two observations are said to belong to the same cluster if their corresponding centroids are equal. Then clustering analysis is formulated to identify a small subset of distinct values of these centroids via solving a penalized regression problem. For more details, see the following two papers.

* Pan Wei, Xiaotong Shen, and Binghui Liu. "Cluster Analysis: Unsupervised Learning via Supervised Learning with a Non-convex Penalty." *The Journal of Machine Learning Research* 14.1 (2013):1865-1889.
* Wu C, Kwon S, Shen X, Pan W. "A New Algorithm and Theory for Penalized Regression-based Clustering." *Journal of Machine Learning Research*. 2016;17(188):1-25.  

```r
library("prclust")
## generate the data
data = matrix(NA,2,100)
data[1,1:50] = rnorm(50,0,0.33)
data[2,1:50] = rnorm(50,0,0.33)
data[1,51:100] = rnorm(50,1,0.33)
data[2,51:100] = rnorm(50,1,0.33)
## set the tunning parameter
lambda1 =1
lambda2 = 3
tau = 0.5
a =PRclust(data,lambda1,lambda2,tau)
a #clustering results
```

## Selecting the tunning parameters 

A bonus with the regression approach to clustering is the potential application of many existing model selection methods for regression or supervised learning to clustering. We propose using generalized cross-validation (GCV). GCV can be regarded as an approximation to leave-one-out cross-validation (CV). Hence, GCV provides an approximately unbiased estimate of the prediction error.

We try with various tuning parameter values, obtaining their corresponding GDFs and thus GCV statistics, then choose the set of the tuning parameters with the minimum GCV statistic.

```r
#case 1
gcv1 = GCV(data,lambda1=1,lambda2=1,tau=0.5,sigma=0.25)
gcv1

#case 2
gcv2 = GCV(data,lambda1=1,lambda2=0.7,tau=0.3,sigma=0.25)
gcv2
```
GCV, while yielding good performance, requires extensive computation and specification of a hyper-parameter perturbation size. Here, we provide an alternative by modifying a stability-based criterion for determining the tuning parameters.

The main idea of the method is based on cross-validation. That is, 

* Randomly partition the entire data set into a training set and a test set with an almost equal size; 
* Cluster the training and test sets separately via PRclust with the same tuning parameters;
* Measure how well the training set clusters predict the test clusters.
*  Try with various tuning parameter values, obtaining their corresponding statbility based statistics average prediction strengths, then choose the set of the tuning parameters with the maximum average prediction stength.

```r
#case 1
stab1 = stability(data,rho=1,lambda=1,tau=0.5,n.times = 10)
stab1

#case 2
stab2 = stability(data,rho=1,lambda=0.7,tau=0.3,n.times = 10)
stab2
```

## Evaluating the clustering results
We provide a function *clusterStat*, which calculates Rand, adjusted Rand, Jaccard index, measuring the agreement between estimated cluster and the truth with a higher value indicating a higher agreement.

```r
truth = c(rep(1,50),rep(2,50))
clustStat(a$group,truth)
```

## Other examples
Some examples have been provided for further illustrating the performance of prclust.

```r

###################################################
#### case 2
###################################################
x1 = -1 + 2*c(0:99)/99
data = matrix(NA,2,200)
data[1,1:100] = x1
data[2,1:100] = (rbinom(100,1,0.5)*2-1)*sqrt(1-x1^2)+runif(100,min = -0.1,max = 0.1)

x2 = -2 + 4*c(0:99)/99
data[1,101:200] = x2
data[2,101:200] = (rbinom(100,1,0.5)*2-1)*sqrt(4-x2^2)+runif(100,min = -0.1,max = 0.1)
## set the tunning parameter
lambda1 =1
lambda2 = 30
tau = 0.6
a =PRclust(data,lambda1,lambda2,tau)
a
## quadratic penalty
lambda1 =1
lambda2 = 1
tau = 0.5
a =PRclust(data,lambda1,lambda2,tau, algorithm ="Quadratic")
a

##################################################
### case 3
##################################################
data = matrix(runif(10*200),10,200)
## set the tunning parameter
lambda1 =1
lambda2 =5
tau = 1
a =PRclust(data,lambda1,lambda2,tau)
a

## quadratic penalty
lambda1 =1
lambda2 =1
tau = 1
a =PRclust(data,lambda1,lambda2,tau, algorithm ="Quadratic")
a
##################################################
### case 4
##################################################
### generate the data set, it's kind of complicate
judge = 1
while(judge != 0)
{
    tempCenter = matrix(rnorm(12,0,5),3,4)
    c1 = tempCenter[,1]
    c2 = tempCenter[,2]
    c3 = tempCenter[,3]
    c4 = tempCenter[,4]

    tempObs1 = 25 + rbinom(1,1,0.5)*25
    tempObs2 = 25 + rbinom(1,1,0.5)*25
    tempObs3 = 25 + rbinom(1,1,0.5)*25
    tempObs4 = 25 + rbinom(1,1,0.5)*25

    Obs = tempObs1 + tempObs2 + tempObs3 + tempObs4
    data = matrix(NA,3,Obs)
    data[1,1:tempObs1] = rnorm(tempObs1,c1[1],1)
    data[2,1:tempObs1] = rnorm(tempObs1,c1[2],1)
    data[3,1:tempObs1] = rnorm(tempObs1,c1[3],1)

    data[1,(tempObs1+1):(tempObs1 + tempObs2)] = rnorm(tempObs2,c2[1],1)
    data[2,(tempObs1+1):(tempObs1 + tempObs2)] = rnorm(tempObs2,c2[2],1)
    data[3,(tempObs1+1):(tempObs1 + tempObs2)] = rnorm(tempObs2,c2[3],1)

    data[1,(tempObs1+tempObs2+1):(tempObs1 + tempObs2 +tempObs3)] = rnorm(tempObs3,c3[1],1)
    data[2,(tempObs1+tempObs2+1):(tempObs1 + tempObs2 +tempObs3)] = rnorm(tempObs3,c3[2],1)
    data[3,(tempObs1+tempObs2+1):(tempObs1 + tempObs2 +tempObs3)] = rnorm(tempObs3,c3[3],1)

    data[1,(tempObs1 + tempObs2 +tempObs3+1):Obs] = rnorm(tempObs4,c4[1],1)
    data[2,(tempObs1 + tempObs2 +tempObs3+1):Obs] = rnorm(tempObs4,c4[2],1)
    data[3,(tempObs1 + tempObs2 +tempObs3+1):Obs] = rnorm(tempObs4,c4[3],1)
    
    a =as.matrix(dist(t(data)))
    if((min(a[1:tempObs1,(tempObs1+1):Obs]) <1 | min(a[(tempObs1+1):(tempObs1+tempObs2),(tempObs1+tempObs2+1):Obs])<1 |min(a[(tempObs1+tempObs2+1):(tempObs1+tempObs2+tempObs3),(tempObs1+tempObs2+tempObs3+1):Obs])<1 )== 0)
    judge =0
}

## set the tunning parameter
lambda1 =1
lambda2 =10
tau =3
a =PRclust(data,lambda1,lambda2,tau)
a

## quadratic penalty
lambda1 =1
lambda2 =1.8
tau =2.3
a =PRclust(data,lambda1,lambda2,tau, algorithm ="Quadratic")
a

#############################################
#### case 5
#############################################
## generate the data set
x1 = -0.5 + c(0:99)/99
data = matrix(NA,3,200)
data[1,1:100] = x1 + rnorm(100,0,0.1)
data[2,1:100] = x1 + rnorm(100,0,0.1)
data[3,1:100] = x1 + rnorm(100,0,0.1)

data[1,101:200] = x1 + 2 + rnorm(100,0,0.1)
data[2,101:200] = x1 + 2 + rnorm(100,0,0.1)
data[3,101:200] = x1 + 2 + rnorm(100,0,0.1)

## set the tunning parameter
lambda1 =1
lambda2 = 1
tau = 0.5
a =PRclust(data,lambda1,lambda2,tau)
a

## quadratic penalty
lambda1 =1
lambda2 = 0.45
tau = 0.35
a =PRclust(data,lambda1,lambda2,tau)
a

#############################################
### case 6
############################################
data = matrix(NA,2,150)
data[1,1:50] =1.5*sin(2*pi*(30+c(0:49)*5)/360)
data[2,1:50] =1.5*cos(2*pi*(30+c(0:49)*5)/360) + runif(50,-0.025,0.025)

data[1,51:100] = rnorm(50,0,0.1)
data[2,51:100] = rnorm(50,0,0.1)

data[1,101:150] = rnorm(50,0.8,0.1)
data[2,101:150] = rnorm(50,0,0.1)
#plot(data[1,],data[2,])
## set the tunning parameter
lambda1 =1
lambda2 = 1
tau = 0.35
a =PRclust(data,lambda1,lambda2,tau)
a

# quadratic penalty
## set the tunning parameter
lambda1 =1
lambda2 = 0.5
tau = 0.4
a =PRclust(data,lambda1,lambda2,tau, algorithm ="Quadratic")
a
```

