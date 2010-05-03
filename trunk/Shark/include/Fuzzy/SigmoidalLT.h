
/**
 * \file SigmoidalLT.h
 *
 * \brief LinguisticTerm with sigmoidal membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SIGMOIDALLT_H
#define SIGMOIDALLT_H

// #include <ZNminmax.h>

#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/SigmoidalFS.h>

/**
 * \brief LinguisticTerm with sigmoidal membership function.
 * 
 * This class implements a LinguisticTerm with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{1 + e^{-a(x-b)}}
 * \f]
 * 
 * <img src="../images/SigmoidalFS.png"> 
 * 
 */
class SigmoidalLT: public LinguisticTerm, public SigmoidalFS {
public:
	
	//            SigmoidalLT(double paramC,double paramOffset);

    /**
	* \brief Constructor.
	*
	* @param name the name
	* @param parent the associated linguistic variable
    * @param paramC scale factor for y-axis
    * @param paramOffset position of inflection point
	*/	
	SigmoidalLT(const std::string name,
	            const RCPtr<LinguisticVariable>& parent,
	            double                           paramC = 1,
	            double                           paramOffset = 0);

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMin() const {
		return(std::max(SigmoidalFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMax() const {
		return(std::min(SigmoidalFS::getMax(), parent->getUpperBound()));
	};

};


#endif
