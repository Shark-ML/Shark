/**
 * \file GeneralizedBellLT.h
 *
 * \brief LinguisticTerm with a generalized bell-shaped membership function.
 * 
 * \authors Thomas Voß
 */

/* $log$ */



#ifndef __GENERALIZEDBELLLT_H__
#define __GENERALIZEDBELLLT_H__

#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/GeneralizedBellFS.h>

/**
 * \brief LinguisticTerm with a generalized bell-shaped membership function.
 * 
 * This class implements a LinguisticTerm with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{1+ (\frac{x-center}{width})^{2*slope}}
 * \f]
 * <img src="../images/GeneralizedBellFS.png"> 
 */
class GeneralizedBellLT : public LinguisticTerm, public GeneralizedBellFS {
public:

/**
 * \brief Constructor.
 *
 * @param name the name of the LinguisticTerm
 * @param parent the associated LinguisticVariable
 * @param slope the slope
 * @param center the center of the bell-shaped function
 * @param width the width of the bell-shaped function
 * @param scale the scale of the bell-shaped function
 */
	GeneralizedBellLT( const std::string & name,
					   const RCPtr<LinguisticVariable> & parent,
					   double slope = 1.0,
					   double center = 0.0,
					   double width = 1.0,
					   double scale = 1.0
					   );


   /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMin() const {
		return(std::max(GeneralizedBellFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
	inline double         getMax() const {
		return(std::min(GeneralizedBellFS::getMax(), parent->getUpperBound()));
	};


};

#endif // __GENERALIZEDBELLLT_H__

