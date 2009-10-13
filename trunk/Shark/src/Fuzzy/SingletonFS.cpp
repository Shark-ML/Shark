/**
 * \file SingletonFS.cpp
 *
 * \brief FuzzySet with a single point of positive membership
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#include <Fuzzy/SingletonFS.h>

void SingletonFS::setParams(double x,double v,double e) {
	c=x;
	yValue=v;
	epsilon=e;
};
