/**
 * \file ComposedLT.cpp
 *
 * \brief A composed LinguisticTerm
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */




#include <Fuzzy/ComposedLT.h>

ComposedLT::ComposedLT(const std::string name,
                       const RCPtr<LinguisticVariable>& parent ,
                       Operator  o,
                       const RCPtr<FuzzySet>& f1,
                       const RCPtr<FuzzySet>& f2): LinguisticTerm(name,parent),ComposedFS(o,f1,f2) {};

ComposedLT::ComposedLT(const std::string name,
                       const RCPtr<LinguisticVariable>& parent ,
                       Operator  o,
                       const RCPtr<FuzzySet>& f1,
                       const RCPtr<FuzzySet>& f2,
                       double (*userFunction)( double,double ) ):
		LinguisticTerm(name,parent),ComposedFS(o,f1,f2,userFunction) {};
