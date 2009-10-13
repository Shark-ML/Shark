
/**
 * \file InfinityFS.cpp
 *
 * \brief FuzzySet with a step function as membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#include <Fuzzy/InfinityFS.h>

InfinityFS::InfinityFS(bool b,double d,double f):
		positiveInfinity(b),a(d),b(f) {};

double  InfinityFS::mu( double x ) const {
	if (a>b) {
		throw(FuzzyException(1,"Illegal parameters for triangular MF --> a > b"));
	};
	if (positiveInfinity)
		if (x>=b)
			return(1);
		else
			return((x-a)/(b-a)*(a<=x));
	else
		if (x<=a)
			return(1);
		else
			return((b-x)/(b-a)*(x<=b));
}

void InfinityFS::setParams(bool b,double d,double f) {
	positiveInfinity = b;
	this->a = d;
	this->b =f;
}
