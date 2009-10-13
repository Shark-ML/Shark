/**
 * \file TrapezoidFS.cpp
 *
 * \brief FuzzySet with trapezoid membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */


#include <Fuzzy/TrapezoidFS.h>
#include <algorithm> // for min and max

TrapezoidFS::TrapezoidFS(double p1, double p2, double p3, double p4):
		a(p1), b(p2), c(p3), d(p4) {};

TrapezoidFS::~TrapezoidFS() {};

double  TrapezoidFS::mu(double x) const {
	if (a>b || b>c || c>d) {
		throw(FuzzyException(1,"Illegal parameters for trapezoid MF"));
	};
	if (a == b && c == d)
		return(a<=x && x<=d);
	if (a == b)
		return(x<=c?b<=x:(d-x)/(d-c)*(b<=x)*(x<=d));
	if (c == d)
		return(b<=x?x<=c:(x-a)/(b-a)*(a<=x)*(x<=b));
	return((x<b)||(x>c)?std::max(std::min((x-a)/(b-a), (d-x)/(d-c)), 0.0):1.0);
};



void TrapezoidFS::setParams(double  w, double x, double y, double z) {
	a=w, b=x, c=y, d=z;
};
