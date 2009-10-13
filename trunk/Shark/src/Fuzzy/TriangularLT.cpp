/**
 * \file TriangularLT.cpp
 *
 * \brief LinguisticTerm with triangular membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */


#include <Fuzzy/TriangularLT.h>

TriangularLT::TriangularLT(const std::string name,
                           const RCPtr<LinguisticVariable>& parent,
                           double                           p1,
                           double                           p2,
                           double                           p3):
		LinguisticTerm(name,parent),TriangularFS(p1,p2,p3) {}
