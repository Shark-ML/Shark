
/**
 *
 * \brief LinguisticTerm with a step function as membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */
#ifndef SHARK_FUZZY_INFINITYLT_H
#define SHARK_FUZZY_INFINITYLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/InfinityFS.h>

namespace shark {

/**
   * \brief LinguisticTerm with a step function as membership function.
   *
   * The support of this LinguisticTerm reaches to either positive infinity or negative
   * infinity. The corresponding menbership function is discribed by two dedicated points a and b.
   * If the support reaches positive infinity, the membership function is 0 for values smaller than a,
   * raises constantly to 1 between a and b, and stays 1 for values greater than b. If the support 
   * reaches negantive infinity, the membership function is 1 for values smaller than a, falls 
   * constantly to 0 between a and b, and is 0 for values greater than b. 
   *
   * For positive Infinity:
   * \f[
   *      \mu(x) = \left\{\begin{array}{ll} 0, & x < a \\ 
   *      \frac{1}{b-a}(x-a), & a \le x \le b \\ 
   *      1, & x > b\end{array}\right.
   * \f]
   * <img src="../images/InfinityFS.png"> 
   * 
   */
class InfinityLT : public LinguisticTerm, public InfinityFS {
public:

    /**
     * \brief Construnctor.
     * 
     * @param name the name
     * @param parent the associated linguistic variable 
     * @param infType decides whether the support reaches to positive or negative infinity
     * @param a point where membership function starts to raise/fall (positiveInfinity: true/false)
     * @param b point where membership stops to raise/fall (positiveInfinity: true/false)
     */
    InfinityLT(const std::string & name = "InfinityLT",
               const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
               InfinityType infType = POSITIVE_INFINITY,
               double a = 0.,
               double b = 1. ) : LinguisticTerm( name ),
        InfinityFS( infType, a, b ) {
        setLinguisticVariable(parent);
    }


    // overloaded operator () - the mu function
    //inline double operator()(double x) const;

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( std::max( InfinityFS::min(), m_parent->lowerBound() ) );
    }

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
        return( std::min( InfinityFS::max(), m_parent->upperBound() ) );
    }
};

}
#endif
