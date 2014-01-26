
/**
 *
 * \brief FuzzySet with constant membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef SHARK_FUZZY_CONSTANTFS_H
#define SHARK_FUZZY_CONSTANTFS_H

#include <shark/Fuzzy/FuzzySet.h>

#include <limits>

namespace shark {

/**
   * \brief FuzzySet with constant membership function.
   * 
   * This class implements a FuzzySet with constant membership function.
   * 
   * <img src="../images/ConstantFS.png"> 
   * 
   */
class ConstantFS: virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     *
     * @param x the constant value of the membership function
     */
    ConstantFS( double x = 0. ) : m_c( x ) {}

    /**
     * \brief Defuzzification of the fuzzy set.  
     */
    inline double defuzzify() const {
        return( 0 );
    };

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( -std::numeric_limits<double>::max() );
    } ;

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
        return(std::numeric_limits<double>::max());
    };

    /**
     * \brief Sets the parameter of the fuzzy set.
     *
     * @param x the constant value of the membership function
     */
    /*inline void        setParams(double x){
      c=x;
      };*/

private:
    double mu(double x) const {
        return( m_c );
    };
    double m_c;
};

ANNOUNCE_FUZZY_SET( ConstantFS, FuzzySetFactory );
}
#endif
