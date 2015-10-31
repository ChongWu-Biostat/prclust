\name{PRclust}
\alias{PRclust}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Find the solution of penalized regression-based clustering.
}
\description{
	Clustering is unsupervised and exploratory in nature. Yet, it can be performed through penalized regression with grouping pursuit
. Prclust helps us peform penalized regression-based clustering with various loss functions and grouping penalities via two algorithm (DC-ADMM and quadratic penalty).
}
\usage{
PRclust(data, lambda1, lambda2, tau, 
	loss.method = c("quadratic","lasso"), 
	group.method = c("gtlp","lasso","SCAD","MCP"), 
	algorithm = c("ADMM","Quadratic"), epsilon=0.001)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
	 input matrix, of dimension nvars x nobs; each column is an observation vector.
}
   \item{lambda1}{
     Tuning parameter or step size: lambda1, typically set at 1 for quadratic penalty based algorithm; 0.4 for revised ADMM.
  }
  \item{lambda2}{
  	 Tuning parameter: lambda2, the magnitude of grouping penalty.
}
  \item{tau}{
  	 Tuning parameter: tau, related to grouping penalty.
}

\item{loss.method }{
      The loss method. "lasso" stands for \eqn{L_1} loss function, while "quadratic" stands for the quadratic loss function.  
}
  \item{group.method}{
    character: may be abbreviated. "gtlp" means generalized group lasso is used for grouping penalty. "lasso" means lasso is used for grouping penalty. "SCAD" and "MCP" are two other non-convex penalty.
}
  \item{algorithm}{
	character: may be abbreviated. The algorithm to use for finding the solution. The default algorithm is "ADMM", which stands for the new algorithm we developed.
}
\item{epsilon}{The stopping critetion parameter. The default is 0.001.
}
}
\details{
Clustering analysis has been widely used in many fields. In the absence of a class label, clustering analysis is also called unsupervised learning. However, penalized regression-based clustering adopts a novel framework for clustering analysis by viewing it as a regression problem. In this method, a novel non-convex penalty for grouping pursuit was proposed which data-adaptively encourages the equality among some unknown subsets of parameter estimates. This new method can deal with some complex clustering situation, for example, in the presence of non-convex cluster, in which the K-means fails to work, PRclust might perform much better.
}
\value{
The return value is a list. In this list, it contains the following matrix.
 \item{mu}{The centroid of the each observations.}
 \item{theta}{The theta value for the data set, not very useful.}
 \item{group}{The group for each points.}
 \item{count}{The iteration times.}
}
\references{
Pan Wei, Xiaotong Shen, and Binghui Liu. "Cluster Analysis: Unsupervised Learning via Supervised Learning with a Non-convex Penalty." \emph{The Journal of Machine Learning Research} 14.1 (2013):1865-1889.

Chong Wu, Sunghoon Kwon, Xiaotong Shen and Wei Pan. "A new Algorithm and Theory for Penalized Regression-based Clustering", submitted. 
}
\author{
Chong Wu, Wei Pan
}
\note{
Choosing tunning parameter is kind of time consuming job. It is always based on "trials and errors".
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
library("prclust")
# To let you have a better understanding about the power and strength
# of PRclust method, 6 examples in original prclust paper were provided.
################################################
### case 1
################################################
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
a

## quadratic penalty
lambda1 =1
lambda2 = 1
tau = 0.5
a =PRclust(data ,lambda1,lambda2,tau, algorithm ="Quadratic")
a
###################################################
#### case 2
###################################################
x1 = -1 + 2*c(0:99)/99
data = matrix(NA,2,200)
data[1,1:100] = x1
temp = (rbinom(100,1,0.5)*2-1)*sqrt(1-x1^2)
temp = temp +runif(100,min = -0.1,max = 0.1)

data[2,1:100] = temp

x2 = -2 + 4*c(0:99)/99
data[1,101:200] = x2
 temp = (rbinom(100,1,0.5)*2-1)*sqrt(4-x2^2)
 temp = temp +runif(100,min = -0.1,max = 0.1)
data[2,101:200] =temp

## set the tunning parameter
lambda1 =1
lambda2 = 30
tau = 0.6
a =PRclust(data ,lambda1,lambda2,tau)
a
## quadratic penalty
lambda1 =1
lambda2 = 1
tau = 0.5
a =PRclust(data ,lambda1,lambda2,tau, algorithm ="Quadratic")
a

##################################################
### case 3
##################################################
data = matrix(runif(10*200),10,200)
## set the tunning parameter
lambda1 =1
lambda2 =5
tau = 1
a =PRclust(data ,lambda1,lambda2,tau)
a

## quadratic penalty
lambda1 =1
lambda2 =1
tau = 1
a =PRclust(data ,lambda1,lambda2,tau, algorithm ="Quadratic")
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

data.index = (tempObs1+tempObs2+1):(tempObs1 + tempObs2 +tempObs3)
data[1, data.index] = rnorm(tempObs3,c3[1],1)
data[2, data.index] = rnorm(tempObs3,c3[2],1)
data[3, data.index] = rnorm(tempObs3,c3[3],1)

data[1,(tempObs1 + tempObs2 +tempObs3+1):Obs] = rnorm(tempObs4,c4[1],1)
data[2,(tempObs1 + tempObs2 +tempObs3+1):Obs] = rnorm(tempObs4,c4[2],1)
data[3,(tempObs1 + tempObs2 +tempObs3+1):Obs] = rnorm(tempObs4,c4[3],1)
    
a =as.matrix(dist(t(data)))
if((min(a[1:tempObs1,(tempObs1+1):Obs]) <1 | 
min(a[(tempObs1+1):(tempObs1+tempObs2),(tempObs1+tempObs2+1):Obs])<1
| min(a[(tempObs1+tempObs2+1):(tempObs1+tempObs2+tempObs3),
(tempObs1+tempObs2+tempObs3+1):Obs])<1 )== 0)
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
temp = 1.5*cos(2*pi*(30+c(0:49)*5)/360) 
temp = temp + runif(50,-0.025,0.025)
data[2,1:50] = temp
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

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{PRclust}
