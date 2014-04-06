
/**
*
* \brief LinguisticTerm with a bell-shaped (Gaussian) membership function
* 
* \authors Marc Nunkesser
*/


/* $log$ */
#ifndef SHARK_FUZZY_BELLLT_H
#define SHARK_FUZZY_BELLLT_H

#include <shark/Fuzzy/FuzzySets/BellFS.h>
#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <algorithm>
#include <string>

namespace shark {
/**
 * \brief LinguisticTerm with a bell-shaped (Gaussian) membership function
 *
 * This class implements a LinguisticTerm with membership function:
 *
 * \f[
 *      \mu(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-offset)^2}{2\sigma^2}}
 * \f]
 *
 * <img src="../images/BellFS.png">
 *
 */
class BellLT : public LinguisticTerm, public BellFS {
public:

    /**
  * \brief Constructor.
  *
  * @param name the name
  * @param parent the associated linguistic variable
  * @param sigma controlls the width of the Gaussian
  * @param offset position of the center of the peak
  * @param scale scales the whole function
  */
    BellLT(const std::string & name = "BellLT",
           const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
           double sigma = 1.,
           double offset = 0.,
           double scale = 1) : LinguisticTerm( name ),
        BellFS( sigma, offset, scale ) {
		setLinguisticVariable(parent);
    }


    /**
  * \brief Defuzzifies the set by returning the Bell's offset if the support is not
  *  bounded by the associated linguistic variable, and by dertermination of the center
  *  of gravity otherwise.
  *
  * @param errRel relative approximation error that is tollerated during numerical integration
  * @param recursionMax max. depth of recursion during numerical integration (i.e. max. \f$2^n\f$ steps)
  */
    double defuzzify(double errRel = FuzzySet::RELATIVE_ERROR(), std::size_t recursionMax = FuzzySet::RECURSION_MAX() ) const {
        return(
                    (BellFS::min() >= m_parent->lowerBound() ) && ( BellFS::max() <= m_parent->lowerBound() ) ?
                        BellFS::defuzzify() :
                        FuzzySet::defuzzify(m_parent->lowerBound(), m_parent->upperBound(), errRel, recursionMax)
                        );
    }

    /**
  * \brief Returns the lower boundary of the support.
  *
  * @return the min. value for which the FuzzySet is nonzero (or exceeds a
  * given threshold)
  */
    inline double min() const {
        return( std::max( BellFS::min(), m_parent->lowerBound() ) );
    };

    /**
  * \brief Returns the upper boundary of the support.
  *
  * @return the max. value for which the FuzzySet is nonzero (or exceeds a
  * given threshold)
  */
    inline double max() const {
        return( std::min( BellFS::max(), m_parent->upperBound () ) );
    };
};

}
#endif
