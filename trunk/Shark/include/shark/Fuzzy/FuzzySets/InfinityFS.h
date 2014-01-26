/**
 *
 * \brief FuzzySet with a step function as membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef SHARK_FUZZY_INFINITYFS_H
#define SHARK_FUZZY_INFINITYFS_H

#include <shark/Fuzzy/FuzzySet.h>

#include <limits>

namespace shark {

/**
   * \brief FuzzySet with a step function as membership function. 
   * 
   * The support of this FuzzySet reaches to either positive infinity or negative
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
   *
   *   
   */
class InfinityFS : virtual public FuzzySet {
public:

    enum InfinityType {
        POSITIVE_INFINITY,
        NEGATIVE_INFINITY
    };

    /**
     * \brief Construnctor.
     * 
     * @param infType decides whether the support reaches to positive negative infinity
     * @param a point where membership function starts to raise/fall (positiveInfinity: true/false)
     * @param b point where membership function stops to raise/fall (positiveInfinity: true/false)
     */
    InfinityFS( InfinityType infType = POSITIVE_INFINITY, double a = 0., double b = 1. ) : m_infType( infType ),
        m_a( a ),
        m_b( b ) {
    }

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( m_infType == POSITIVE_INFINITY ? m_a : -std::numeric_limits<double>::max() );
    };

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
        return( m_infType == POSITIVE_INFINITY ? std::numeric_limits<double>::max() : m_b );
    };


protected:
    double mu(double x) const {
        double result;
        switch( m_infType ) {
        case POSITIVE_INFINITY:
            if( x >= m_b )
                result = 1;
            else
                result = (x - m_a)/(m_b - m_a) * (m_a <= x);
            break;
        case NEGATIVE_INFINITY:
            if( x <= m_a )
                result = 1;
            else
                result = (m_b - x)/(m_b - m_a) * (x <= m_b);
            break;
        }
        return( result );
    }
    InfinityType m_infType;
    double m_a;
    double m_b;

};

ANNOUNCE_FUZZY_SET( InfinityFS, FuzzySetFactory );
}
#endif
