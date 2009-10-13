/**
 * \file InfinityLT.cpp
 *
 * \brief LinguisticTerm with a step function as membership function
 * 
 * \authors Marc Nunkesser
 */

#include <Fuzzy/InfinityLT.h>

InfinityLT::InfinityLT(const std::string name,
                       const RCPtr<LinguisticVariable>& parent,
                       bool                             p1,
                       double                           p2,
                       double                           p3):
		LinguisticTerm(name,parent),InfinityFS(p1,p2,p3) {}


