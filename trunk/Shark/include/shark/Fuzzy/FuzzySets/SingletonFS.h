/**
 *
 * \brief FuzzySet with a single point of positive membership
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SHARK_FUZZY_SINGLETONFS_H
#define SHARK_FUZZY_SINGLETONFS_H

#include <shark/Fuzzy/FuzzySet.h>
#include <cmath>

namespace shark {

/**
   * \brief FuzzySet with a single point of positive membership.
   * 
   * The membership function of a SingletonFS takes value 1 only at a given point
   * and value 0 everywhere else.
   */
class SingletonFS : virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     * 
     * @param x position of membership point
     * @param yValue value of μ(x)
     * @param eps range of membership around x
     */

    SingletonFS( double x = 0.0,
		 double yValue = 1.0,
		 double eps = 1E-5
                 ): m_c( x ), m_yValue( yValue ),m_epsilon( eps ) {};

    /**
     * \brief Defuzzification
     *
     */
    double defuzzify() const{
        return( m_c );
    }

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or 
     * exceeds a given threshold)
     */
    double min() const {
        return( m_c );
    }

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
        return( m_c );
    }

protected:
    double mu( double x) const {
        return( ::fabs(x-m_c) < m_epsilon ? m_yValue : 0.0 );
    };
    
    double m_c; ///< The position at which the singleton is defined (mu(c)=yvalue)
    double m_yValue;
    double m_epsilon; ///< The precision to which the comparison is carried out
};

}
#endif
