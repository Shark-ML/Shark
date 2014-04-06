
/**
 *
 * \brief LinguisticTerm with sigmoidal membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SIGMOIDALLT_H
#define SIGMOIDALLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>
#include <shark/Fuzzy/FuzzySets/SigmoidalFS.h>

namespace shark {
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

    /**
     * \brief Constructor.
     *
     * @param name the name
     * @param parent the associated linguistic variable
     * @param paramC scale factor for y-axis
     * @param paramOffset position of inflection point
     */	
    SigmoidalLT(const std::string & name = "SigmoidalLT",
                const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
                double paramC = 1.,
                double paramOffset = 0. ) : LinguisticTerm( name ),
        SigmoidalFS( paramC, paramOffset ) {
        setLinguisticVariable(parent);
    }

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double min() const {
        return( std::max( SigmoidalFS::min(), m_parent->lowerBound() ) );
    }

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double max() const {
        return( std::min( SigmoidalFS::max(), m_parent->upperBound() ) );
    }

};
}

#endif
