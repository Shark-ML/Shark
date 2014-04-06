
/**
 *
 * \brief FuzzySet with sigmoidal membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */
#ifndef SHARK_FUZZY_SIGMOIDALFS_H
#define SHARK_FUZZY_SIGMOIDALFS_H

#include <shark/Fuzzy/LinguisticTerm.h>

#include <limits>
#include <cmath>
#include <cassert>

namespace shark {

/**
   * \brief FuzzySet with sigmoidal membership function.
   * 
   * This class implements a FuzzySet with membership function:
   * 
   * \f[
   *      \mu(x) = \frac{1}{1 + e^{-a(x-b)}}
   * \f]
   * 
   * <img src="../images/SigmoidalFS.png"> 
   * 
   */
class SigmoidalFS: virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     *
     * @param paramC scale factor for y-axis
     * @param paramOffset position of inflection point
     */	
    SigmoidalFS(double paramC = 1,double paramOffset = 0) : m_c( paramC ), 
        m_offset( paramOffset ),
        m_threshold( 1E-6 ) {
    }

    // overloaded operator () - the mu-function
    // The sigmoidal mf (normalized i.e the max is always 1) is represented
    // by two parameters c and offset:
    // sig(c,offset) = 1/(1+exp(-c(x-offset))
    //

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( m_c <= 0 ? -std::numeric_limits<double>::min() : (-1/m_c)*::log(1/m_threshold-1)+m_offset);
    }

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double max() const {
        return( (-1/m_c)*::log(1/m_threshold-1)+m_offset );
    }

    /**
     * \brief Returns the threshold for which the membership function (for which the x axis
     * is the asymptote) is set to be zero in the numerical intergration.
     *
     * @return the threshold
     */
    inline double threshold() const {
        return( m_threshold );
    };

    /**
     * \brief Sets the threshold for which the membership function (for which the x axis
     * is the asymptote) is set to be zero in the numerical intergration.
     *
     * @param thr the threshold 
     */
    inline void setThreshold( double thr ) {
        m_threshold = thr;
    }

    /**
     * \brief Sets the parameters of the fuzzy set.
     *
     * @param paramC 
     * @param paramOffset 
     */	
    void setParams(double paramC,double paramOffset);

protected:
    double mu(double x) const {
        return( 1 / ( 1+::exp( -m_c*( x-m_offset ) ) ) );
    }
    double m_c;
    double m_offset;
    double m_threshold;
};

}

#endif
