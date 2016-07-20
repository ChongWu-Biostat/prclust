\name{GCV}
\alias{GCV}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Calculate the Generalized Cross-Validation Statistic (GCV)}
\description{
Calculate the generalized cross-validation statistic with generalized degrees of freedom. We try with various tuning parameter values, obtaining their corresponding GDFs and thus GCV statistics, then choose the set of the tuning parameters with the minimum GCV statistic.
}
\usage{
GCV(data,rho,lambda,tau,sigma,B=100,
    loss.function = c("quadratic","L1"),
    grouping.penalty = c("gtlp","tlp"), 
    algorithm = c("DCADMM","Quadratic"),
    abs_res = 0.5,rel_res = 0.5)
}
%- maybe also 'usage' for other objects documented here.
\arguments{

 \item{data}{
	 Input matrix. Each column is an observation vector.
}
   \item{rho}{
     Tuning parameter or step size: rho, typically set at 1 for quadratic penalty based algorithm; 0.4 for DC-ADMM. (Note that rho is the lambda1 in quadratic penalty based algorithm.)
  }
  \item{lambda}{
  	 Tuning parameter: lambda, the magnitude of grouping penalty.
}
  \item{tau}{
  	 Tuning parameter: tau, a nonnegative tuning parameter controll ing the trade-off between the model fit and the number of clusters.
}
  \item{sigma}{
  	 The perturbation size.
  }
  \item{B}{
  	 The Monte Carlo time. The defualt value is 100.
  }

\item{loss.function }{
      The loss function. "L1" stands for \eqn{L_1} loss function, while "quadratic" stands for the quadratic loss function.  
}
  \item{grouping.penalty}{
    Grouping penalty: "gtlp" is the group truncated lasso peanlty, while "tlp" is the truncated lasso. Note that when tau is large, "gtlp" and "tlp" are the $L_2$ penalty and lasso penalty, respectively.
}
  \item{algorithm}{
  Two algorithms for PRclust. "DC-ADMM" and "Quadratic" stand for the DC-ADMM and quadratic penalty based criterion respectively. "DC-ADMM" is much faster than "Quadratic" and thus recommend it here. 
}
\item{abs_res}{Primal residual, the stopping critetion parameter corresponding to DC-ADMM. The default is 0.5.
}
\item{rel_res}{Dual residual, the stopping critetion parameter corresponding to DC-ADMM. The default is 0.5.
}
}

\details{
A bonus with the regression approach to clustering is the potential application of many existing model selection methods for regression or supervised learning to clustering. We propose using generalized cross-validation (GCV). GCV can be regarded as an approximation to leave-one-out cross-validation (CV). Hence, GCV provides an approximately unbiased estimate of the prediction error.

We use the generalized degrees of freedom (GDF) to consider the data-adaptive nature in estimating the centroids of the observations.

The chosen tuning parameters are the one giving the smallest GCV error.
}
\value{
Return value: the Generalized cross-validation statistic (GCV)
}

\references{
Pan Wei, Xiaotong Shen, and Binghui Liu. "Cluster Analysis: Unsupervised Learning via Supervised Learning with a Non-convex Penalty." \emph{The Journal of Machine Learning Research} 14.1 (2013):1865-1889.
}
\author{
Chong Wu, Wei Pan
}
\examples{
set.seed(1)
library("prclust")
data = matrix(NA,2,50)
data[1,1:25] = rnorm(25,0,0.33)
data[2,1:25] = rnorm(25,0,0.33)
data[1,26:50] = rnorm(25,1,0.33)
data[2,26:50] = rnorm(25,1,0.33)

#case 1
gcv1 = GCV(data,rho=1,lambda=1,tau=0.5,sigma=0.25,B =10)
gcv1

#case 2
gcv2 = GCV(data,rho=1,lambda=0.7,tau=0.3,sigma=0.25,B = 10)
gcv2

# Note that the combination of tuning parameters in case 1 are better than 
# the combination of tuning parameters in case 2 since the value of GCV in case 1 is
# less than the value in case 2.
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{GCV}
