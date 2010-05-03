/**
 * \file SigmoidalFS.cpp
 *
 * \brief FuzzySet with sigmoidal membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */


#include <Fuzzy/SigmoidalFS.h>
#include <cmath>

double            SigmoidalFS::mu(double x) const {
	return(1/(1+exp(-c*(x-offset))));
};



void SigmoidalFS::setThreshold(double t) {
	assert((t>0)&&(t<1));
	threshold = t;
};




void SigmoidalFS::setParams(double paramC, double paramOffset) {
	c=paramC;
	offset=paramOffset;
};
