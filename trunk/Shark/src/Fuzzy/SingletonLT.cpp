/**
 * \file SingletonLT.cpp
 *
 * \brief LinguisticTerm with a single point of positive membership
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#include <Fuzzy/SingletonLT.h>

SingletonLT::SingletonLT(const std::string                name,
                         const RCPtr<LinguisticVariable>& parent,
                         double                           p1):
		LinguisticTerm(name,parent),SingletonFS(p1) {};


double SingletonLT::defuzzify() const {
	double result = SingletonFS::defuzzify();
	result = std::min( result, parent->getUpperBound() );
	result = std::max( result, parent->getLowerBound() );
	return result;
};
