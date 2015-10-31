\name{clusterStat}
\alias{clusterStat}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
External evaluation of cluster results
}
\description{
Suppose we know the true cluster results. This function provides Rand, adjusted Rand, Jaccard index to measure the quality of a cluster results.
}
\usage{
clusterStat(trueGroup, group)
}
\arguments{
  \item{trueGroup}{The true cluster results.
}
  \item{group}{The estimated cluster results, not neccessary calculating by PRclust.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
The return value is a "clusterStat" class, providing the following information.
 \item{Rand}{Rand Index}
 \item{AdjustedRand}{Adjusted Rand Index}
 \item{Jaccard}{Jaccard Index}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Chong Wu
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
	a <- rep(1:3,3)
	a
	b <- rep(c(4:6),3)
	b
	clusterStat(a,b)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
