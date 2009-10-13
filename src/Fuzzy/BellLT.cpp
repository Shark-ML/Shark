
/**
 * \file BellLT.cpp
 *
 * \brief LinguisticTerm with a bell-shaped (Gaussian) membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */
#include <Fuzzy/BellLT.h>

BellLT::BellLT(const std::string name,
               const RCPtr<LinguisticVariable>& parent,
               double sigma,
               double offset,
               double scale):
		LinguisticTerm(name,parent),BellFS(sigma,offset,scale) {};

double BellLT::defuzzify( double errRel, int recursionMax ) const
// is the bell shaped FS entirely in the support given by the Linguistic Variable? If so, the simple defuzzification of bellFS can be used
{
	return( ( BellFS::getMin() >= parent->getLowerBound() ) && ( BellFS::getMax() <= parent->getLowerBound() ) ? 
               BellFS::defuzzify() : FuzzySet::defuzzify(parent->getLowerBound(), parent->getUpperBound(), errRel, recursionMax) );
}

