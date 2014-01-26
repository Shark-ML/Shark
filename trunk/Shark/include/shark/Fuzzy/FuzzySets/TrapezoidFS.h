/**
 *
 * \brief FuzzySet with trapezoid membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SHARK_FUZZY_TRAPEZOIDFS_H
#define SHARK_FUZZY_TRAPEZOIDFS_H

#include <shark/Fuzzy/FuzzySet.h>

namespace shark {

/**
   * \brief FuzzySet with trapezoid membership function.
   * 
   * This class implements a FuzzySet with a trapezoid membership function.
   * A trapezoid membership function is definded by four values a,b,c,d. 
   * Points smaller than a and bigger than d have the value 0. The function
   * increases constantly to the value 1 between a and b, stays 1 between b 
   * and c, and decreases constantly between c and d.
   * 
   * \f[
   * 		\mu(x) = \left\{\begin{array}{ll} 0 & x < a \\ 
   *      \frac{1}{b-a}(x-a) & a \le x < b \\
   *      1 & b \leq x < c \\
   * 		\frac{1}{d-c}(d-x) & c \le x < d \\
   *      0 & x \geq d\end{array}\right.
   * \f]
   * 
   * <img src="../images/TrapezoidFS.png">
   */
class TrapezoidFS: virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     *
     * @param a the minimal value for which the membership function is nonzero
     * @param b the value to which the membership function increases to value 1 
     * @param c the value to which the membership function stays 1
     * @param d the maximal value for which the membership function is nonzero
     */
    TrapezoidFS( double a = 0.,
                 double b = 1.,
                 double c = 2.,
                 double d = 3. ) : m_a( a ),
        m_b( b ),
        m_c( c ),
        m_d( d ) {

	if( a > b || b > c || c > d )
            throw( shark::Exception( "TrapezoidFS: Invalid arguments.", __FILE__, __LINE__ ) );

    }
    
    virtual ~TrapezoidFS() {}

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    virtual double min() const {
        return( m_a );
    };

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    virtual double max() const {
        return( m_d );
    };



private:
    // overloaded operator () - the mu-function
    // The trapezoid mf (normalized i.e the max is always 1) is represented
    // by three points: the first nonzero value a, value b so that mu(b)=1
    // and the last nonzero value c.
    double mu(double x) const {
        if (m_a == m_b && m_c == m_d)
            return( m_a <= x && x <= m_d );
        if( m_a == m_b )
            return( x <= m_c ? m_b <= x : ( m_d - x )/( m_d - m_c )*( m_b <= x )*( x <= m_d) );
        if( m_c == m_d )
            return( m_b <= x ? x <= m_c : ( x - m_a )/( m_b - m_a ) * ( m_a <= x )*( x <= m_b ) );
        return( (x < m_b) || (x > m_c) ? std::max ( std::min( (x - m_a)/(m_b - m_a), (m_d - x)/(m_d - m_c) ), 0.0 ) : 1.0 );
    }

    double m_a;
    double m_b;
    double m_c;
    double m_d;

};
ANNOUNCE_FUZZY_SET( TrapezoidFS, FuzzySetFactory );
}
#endif
