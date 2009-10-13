/**
 * \file BellFS.cpp
 *
 * \brief FuzzySet with a bell-shaped (Gaussian) membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */
#include <Fuzzy/BellFS.h>

const double BellFS::factor = 0.5*M_2_SQRTPI*M_SQRT1_2;
const double BellFS::factor2 = 2*M_SQRT2/M_2_SQRTPI;

BellFS::BellFS(double s, double o, double c):
		FuzzySet(),
		sigma(s),
		offset(o),
		scale(c),
		threshold(1E-6) {
	setMaxMin();
};


void BellFS::setThreshold(double t) {
	assert((t>0)&&(t<1));
	threshold = t;
	setMaxMin();
};

void BellFS::setParams(double s, double o, double c) {
	sigma = s;
	offset = o;
	scale = c;
	setMaxMin();
};


double BellFS::mu( double x ) const {
	//return( exp( (-(x-offset)*(x-offset))/(2*sigma*sigma) ) );
	return(scale/sigma*factor*exp(-pow((x-offset),2)/(2*sigma*sigma)));
};


void              BellFS::setMaxMin()
// Calculate where bellFS becomes greater than the threshold value
// by setting bell = threshold
// which yields x=+-sigma*sqrt(2ln(threshold*sigma/scale*factor2))+offset
{
	mn = offset + sigma*sqrt(2*log(threshold));
	mx = offset - sigma*sqrt(2*log(threshold));
	
	double radicand;
	if ((radicand = log(threshold*sigma/scale*factor2))>0) {
		mn = -std::numeric_limits<double>::max();
		mx = std::numeric_limits<double>::max();
	} else {
		const double temp = sqrt(-2*radicand)*sigma;
		mx = offset+temp;
		mn = offset-temp;
	}
};
