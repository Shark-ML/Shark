/**
 * \file CustomizedLT.cpp
 *
 * \brief A LinguisticTerm with an user defined mambership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#include <Fuzzy/CustomizedLT.h>

CustomizedLT::CustomizedLT(
    const std::string               name,
    const RCPtr<LinguisticVariable>& parent,
    double (*userFunction)(double),
    double                     min,
    double                     max):
		LinguisticTerm(name,parent),CustomizedFS(userFunction,min,max) {}
