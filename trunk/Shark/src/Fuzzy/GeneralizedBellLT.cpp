/**
 * \file GeneralizedBellLT.cpp
 *
 * \brief LinguisticTerm with a generalized bell-shaped membership function.
 * 
 * \authors Thomas Voß
 */

/* $log$ */


#include <Fuzzy/GeneralizedBellLT.h>

GeneralizedBellLT::GeneralizedBellLT( const std::string & name, 
									  const RCPtr<LinguisticVariable> & parent,
									  double slope, 
									  double center, 
									  double width, 
									  double scale ) : LinguisticTerm( name, parent ),
GeneralizedBellFS( slope, center, width, scale ) {
}
