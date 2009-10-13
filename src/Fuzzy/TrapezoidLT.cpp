/**
 * \file TrapezoidLT.cpp
 *
 * \brief LinguisticTerm with trapezoid membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */


#include <Fuzzy/TrapezoidLT.h>

TrapezoidLT::TrapezoidLT(const std::string name,
                         const RCPtr<LinguisticVariable>& parent,
                         double                           p1,
                         double                           p2,
                         double                           p3,
                         double                           p4):
		LinguisticTerm(name,parent),TrapezoidFS(p1,p2,p3,p4) {}
