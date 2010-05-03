/**
 * \file TriangularFS.cpp
 *
 * \brief FuzzySet with triangular membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */


/* $log$ */

#include <Fuzzy/TriangularFS.h>
#include <algorithm> // for min and max

TriangularFS::TriangularFS(double p1, double p2, double p3):
		a(p1), b(p2), c(p3) {};

TriangularFS::~TriangularFS() {};

double  TriangularFS::mu(double x) const {
	if (a>b) {
		throw(FuzzyException(1,"Illegal parameters for triangular MF --> a > b"));
	};
	if (b>c) {
		throw(FuzzyException(2,"Illegal parameters for triangular MF --> b > c"));
	};
	if (a == b && b == c)
		return(x == a);
	if (a == b)
		return((c-x)/(c-b)*(b<=x)*(x<=c));
	if (b == c)
		return((x-a)/(b-a)*(a<=x)*(x<=b));
	return(std::max(std::min((x-a)/(b-a), (c-x)/(c-b)), 0.0));


};


void TriangularFS::setParams(double x, double y, double z) {
	a=x;
	b= y;
	c=z;
};


