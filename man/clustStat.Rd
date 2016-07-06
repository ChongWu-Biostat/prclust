\name{clustStat}
\alias{clustStat}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
External Evaluation of Cluster Results
}
\description{
Suppose we know the true cluster results beforehand. clusterStat provides Rand, adjusted Rand, Jaccard index to measure the quality of a cluster results.
}
\usage{
clustStat(trueGroup, group)
}
\arguments{
  \item{trueGroup}{The true cluster results.}
  \item{group}{The estimated cluster results, not neccessary calculating by PRclust.}
}

\value{
The return value is a "clusterStat" class, providing the following information.
 \item{Rand}{Rand Index}
 \item{AdjustedRand}{Adjusted Rand Index}
 \item{Jaccard}{Jaccard Index}
%% ...
}

\author{
Chong Wu
}
\examples{
	a <- rep(1:3,3)
	a
	b <- rep(c(4:6),3)
	b
	clustStat(a,b)
}

\keyword{clustStat}