/**
 *
 * \brief LinguisticTerm with a generalized bell-shaped membership function.
 * 
 * \authors Thomas Voß
 */

/* $log$ */
#ifndef SHARK_FUZZY_GENERALIZEDBELLLT_H
#define SHARK_FUZZY_GENERALIZEDBELLLT_H

#include <shark/Fuzzy/LinguisticTerm.h>

#include <shark/Fuzzy/FuzzySets/GeneralizedBellFS.h>

namespace shark {
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
    GeneralizedBellLT( const std::string & name = "GeneralizedBellLT",
                       const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
		       double slope = 1.0,
		       double center = 0.0,
		       double width = 1.0,
		       double scale = 1.0
                       ) : LinguisticTerm( name ),
        GeneralizedBellFS( slope, center, width, scale ) {
        setLinguisticVariable(parent);
    }


    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( std::max( GeneralizedBellFS::min(), m_parent->lowerBound() ) );
    }

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
    double max() const {
        return( std::min( GeneralizedBellFS::max(), m_parent->upperBound() ) );
    }
};

}
#endif

