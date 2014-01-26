
/**
 *
 * \brief LinguisticTerm with trapezoid membership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SHARK_FUZZY_TRAPEZOIDLT_H
#define SHARK_FUZZY_TRAPEZOIDLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/TrapezoidFS.h>

namespace shark {

/**
   * \brief LinguisticTerm with trapezoid membership function.
   * 
   * This class implements a LinguisticTerm with a trapezoid membership function.
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
   * 
   */
class TrapezoidLT: public LinguisticTerm, public TrapezoidFS {
public:

    /**
     * \brief Constructor.
     *
     * @param name the name
     * @param parent the associated linguistic variable
     * @param a the minimal value for which the membership function is nonzero
     * @param b the value to which the membership function increases to value 1 
     * @param c the value to which the membership function stays 1
     * @param d the maximal value for which the membership function is nonzero
     */
    TrapezoidLT(const std::string & name = "TrapezoidLT",
                const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
                double a = 0.,
                double b = 1.,
                double c = 2.,
                double d = 3. ) : LinguisticTerm( name ),
        TrapezoidFS( a, b, c, d ) {
        setLinguisticVariable(parent);
    }

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double min() const {
        return( std::max( TrapezoidFS::min(), m_parent->lowerBound() ) );
    }

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double max() const {
        return( std::min( TrapezoidFS::max(), m_parent->upperBound() ) );
    }

};

ANNOUNCE_LINGUISTIC_TERM( TrapezoidLT, LinguisticTermFactory );
}

#endif
