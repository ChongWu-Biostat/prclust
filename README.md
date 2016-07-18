# prclust  [![License](http://img.shields.io/badge/license-GPL%20%28%3E=%202%29-brightgreen.svg?style=flat)](http://www.gnu.org/licenses/gpl-2.0.html) [![CRAN](http://www.r-pkg.org/badges/version/prclust)](http://cran.rstudio.com/package=prclust) [![Downloads](http://cranlogs.r-pkg.org/badges/prclust?color=brightgreen)](http://www.r-pkg.org/pkg/prclust)

**prclust** is a new R package that makes it incredibly easy to use penalized regression based clustering method with R. 

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
* Chong Wu, Sunghoon Kwon, Xiaotong Shen and Wei Pan. "A new Algorithm and Theory for Penalized Regression-based Clustering", submitted. 

```
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

```
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

```
#case 1
stab1 = stability(data,rho=1,lambda=1,tau=0.5,n.times = 10)
stab1

#case 2
stab2 = stability(data,rho=1,lambda=0.7,tau=0.3,n.times = 10)
stab2
```

## Evaluating the clustering results
We provide a function *clusterStat*, which calculates Rand, adjusted Rand, Jaccard index, measuring the agreement between estimated cluster and the truth with a higher value indicating a higher agreement.

```
truth = c(rep(1,50),rep(2,50))
clustStat(a$group,truth)
```
