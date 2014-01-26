
/**
 *
 * \brief LinguisticTerm with triangular membership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */
#ifndef SHARK_FUZZY_TRIANGULARLT_H
#define SHARK_FUZZY_TRIANGULARLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/TriangularFS.h>

namespace shark {
/**
   * \brief LinguisticTerm with triangular membership function.
   * 
   * This class implements a LinguisticTerm with a triangular membership function.
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
class TriangularLT : public LinguisticTerm, public TriangularFS {
public:

    /**
     * \brief Constructor.
     *
     * @param name the name 
     * @param parent the associated linguistic variable
     * @param a the minimal value for which the membership function is nonzero
     * @param b the value for which the membership function has value 1
     * @param c the maximal value for which the membership function is nonzero
     */
    TriangularLT(const std::string & name = "TriangularLT",
                 const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
                 double a = 0.,
                 double b = 1.,
                 double c = 2. ) : LinguisticTerm( name ),
        TriangularFS( a, b, c ) {
        setLinguisticVariable(parent);
    }
    


    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double min() const {
        return( std::max( TriangularFS::min(), m_parent->lowerBound() ) );
    };

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
    inline double max() const {
        return( std::min( TriangularFS::max(), m_parent->upperBound() ) );
    };

};

ANNOUNCE_LINGUISTIC_TERM( TriangularLT, LinguisticTermFactory );
}
#endif
