/**
 *
 * \brief FuzzySet with triangular membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */


#ifndef SHARK_FUZZY_TRIANGULARFS_H
#define SHARK_FUZZY_TRIANGULARFS_H

#include <shark/Fuzzy/FuzzySet.h>

namespace shark {
/**
   * \brief FuzzySet with triangular membership function.
   * 
   * This class implements a FuzzySet with a triangular membership function.
   * A triangular mambership function has a triangular shape with a
   * maximum value of 1 at a certain point b. 
   *
   * \f[
   * 		\mu(x) = \left\{\begin{array}{ll} 0 & x < a \\ 
   *      \frac{1}{b-a}(x-a) & a \le x < b \\
   * 		\frac{1}{c-b}(c-x) & b \le x < c \\
   *      0 & x \geq c\end{array}\right.
   * \f]
   *  
   * <img src="../images/TriangularFS.png">
   * 
   */
class TriangularFS: virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     *
     * @param a the minimal value for which the membership function is nonzero
     * @param b the value for which the membership function has value 1
     * @param c the maximal value for which the membership function is nonzero
     */
    TriangularFS( double a = 0., double b = 1.,double c = 2. ) : m_a( a ),
        m_b( b ),
        m_c( c ) {
    }
    
    /**
     * \brief Destructor.	
     */	
    virtual ~TriangularFS() {}

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
        return( m_c );
    };
    
protected:
    // overloaded operator () - the mu-function
    // The triangular mf (normalized i.e the max is always 1) is represented
    // by three points: the first nonzero value a, value b so that mu(b)=1
    // and the last nonzero value c.
    
    double mu(double x) const {
        if( m_a == m_b && m_b == m_c ) // ToDo: check with boost
            return(x == m_a);
        if( m_a == m_b )
            return((m_c-x)/(m_c-m_b)*(m_b<=x)*(x<=m_c));
        if (m_b == m_c)
            return((x-m_a)/(m_b-m_a)*(m_a<=x)*(x<=m_b));

        return(std::max(std::min((x-m_a)/(m_b-m_a), (m_c-x)/(m_c-m_b)), 0.0));
    }

    double m_a;
    double m_b;
    double m_c;

};
}
#endif
