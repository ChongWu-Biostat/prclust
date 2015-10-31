\name{prclust-package}
\alias{prclust-package}
\alias{prclust}
\docType{package}
\title{
Penalized Regression Based Cluster Method
}
\description{
Clustering analysis is widely used in many fields. Traditionally clustering is regarded as unsupervised learning for its lack of a class label or a quantitative response variable, which in contrast is present in supervised learning such as classification and regression. Here we formulate clustering as penalized regression with grouping pursuit. In addition to the novel use of a non-convex group penalty and its associated unique operating characteristics in the proposed clustering method, a main advantage of this formulation is its allowing borrowing some well established results in classification and regression, such as model selection criteria to select the number of clusters, a difficult problem in clustering analysis. In particular, we propose using the generalized cross-validation (GCV) based on generalized degrees of freedom (GDF) to select the number of clusters. 

This package provides a user-friendly way to use this new cluster method including GCV to select the tuning parameters.
}
\details{
\tabular{ll}{
Package: \tab prclust\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2015-02-21\cr
License: \tab GPL-2 | GPL-3\cr
}
PRclust: get the solution of penalized regression-based clustering method.Very simple to use. Accepts matrix data for classification, and selects the "best" tuning parameter through generalized cross validation with generalized degree of freedom. Only 3 functions:
\code{PRclust}\cr
\code{GCV}\cr
\code{clusterStat}
}
\author{
Chong Wu, Wei Pan\cr
Maintainer: Chong Wu <chongwu@umn.edu>
}
\references{
Pan Wei, Xiaotong Shen, and Binghui Liu. "Cluster Analysis: Unsupervised Learning via Supervised Learning with a Non-convex Penalty." \emph{The Journal of Machine Learning Research} 14.1 (2013):1865-1889.
}
\keyword{ prclust-package }
\examples{
## In default, we use revised ADMM, a btter algorithm to solve 
## the objective function and get the clustering result.
## generate the data
data = matrix(NA,2,100)
data[1,1:50] = rnorm(50,0,0.33)
data[2,1:50] = rnorm(50,0,0.33)
data[1,51:100] = rnorm(50,1,0.33)
data[2,51:100] = rnorm(50,1,0.33)
## set the tunning parameter
lambda1 =0.4
lambda2 = 1
tau = 0.5
a =PRclust(data,lambda1,lambda2,tau)
a$mu
a$group
}
