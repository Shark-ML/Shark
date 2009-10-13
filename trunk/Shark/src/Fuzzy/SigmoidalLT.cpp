/**
 * \file SigmoidalLT.cpp
 *
 * \brief LinguisticTerm with sigmoidal membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#include <Fuzzy/SigmoidalLT.h>


SigmoidalLT::SigmoidalLT(const std::string        name,
                         const RCPtr<LinguisticVariable>& parent,
                         double                            paramC,
                         double                            paramOffset):
		LinguisticTerm(name,parent), SigmoidalFS(paramC,paramOffset) { };
