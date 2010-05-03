/**
 * \file ConstantLT.cpp
 *
 * \brief Linguistic Term with constant membership function
 * 
 * \authors Asja Fischer and Björn Weghenkel
 */

/* $log$ */

#include <Fuzzy/ConstantLT.h>

ConstantLT::ConstantLT(const std::string name,
	      const RCPtr<LinguisticVariable>&  parent,
	      double x):
         LinguisticTerm(name,parent), ConstantFS(x) {};
		
