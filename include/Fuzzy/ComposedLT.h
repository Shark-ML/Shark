
/**
 * \file ComposedLT.h
 *
 * \brief A composed LinguisticTerm
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef COMPOSEDLT_H
#define COMPOSEDLT_H

#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/RCPtr.h>


/**
 * \brief A composed LinguisticTerm.
 *
 * A composed LinguisticTerm makes it possible to do some
 * calculations on fuzzy sets, e.g. to connect a constant fuzzy set 
 * and a sigmoid fuzzy set using the minimum fuction. This results
 * in a sigmoid fuzzy set which is cutted at the value of the 
 * constant fuzzy set.
 *
 */
class ComposedLT: public LinguisticTerm, public ComposedFS {
public:
	
	
    /**
	* \brief Constructor.
	* 
  	* @param name the name
 	* @param parent the associated linguistic variable 
 	* @param op the operator used (MIN or MAX) to connect the fuzzy sets
 	* @param f1 the first fuzzy set
 	* @param f2 the second fuzzy set
 	*/
	ComposedLT(const std::string name,
	           const RCPtr<LinguisticVariable>& parent ,
	           Operator op,
	           const RCPtr<FuzzySet>& f1,
	           const RCPtr<FuzzySet>& f2);

    /**
	* \brief Constructor.
	*
  	* @param name the name
 	* @param parent the associated linguistic variable 
 	* @param op the operator used (MIN or MAX) to connect the fuzzy sets
 	* @param f1 the first fuzzy set
 	* @param f2 the second fuzzy set
 	* @param userFunction the funcion connecting both fuzzy sets (a user definded operator 
 	*        which is used instead of MIN or MAX) 
 	*/
	ComposedLT(const std::string name,
	           const RCPtr<LinguisticVariable>& parent,
	           Operator op,
	           const RCPtr<FuzzySet>& f1,
	           const RCPtr<FuzzySet>& f2,
	           double (*userFunction)( double,double ) );

    /**
	* \brief Returns the lower boundary of the support
	* 
	* @return the min. value for which the membership function is nonzero (or exceeds a
	* given threshold)
	*/
	inline double         getMin() const {
		return(std::max(ComposedFS::getMin(), parent->getLowerBound()));
	};

    /**
	* \brief Returns the upper boundary of the support
	* 
	* @return the max. value for which the membership function is nonzero (or exceeds a
	* given threshold)
	*/
	inline double         getMax() const {
	return(std::min(ComposedFS::getMax(), parent->getUpperBound()));
	}

	// overloaded operator () - the mu function
	//inline double     operator()(double x) const;
};


#endif
