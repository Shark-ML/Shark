/**
 *
 * \brief A FuzzySet with an user defined mambership function
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef SHARK_FUZZY_CUSTOMIZED_H
#define SHARK_FUZZY_CUSTOMIZED_H

#include <shark/Fuzzy/FuzzySet.h>

namespace shark {
/**
   * \brief A FuzzySet with an user defined mambership function.
   *
   * This class implements a FuzzySet with an user definded membership 
   * function.
   */
class CustomizedFS : virtual public FuzzySet {
public:

    /**
     * \brief Constructor.
     *
     * @param userFunction membership function defined by the user
     * @param min the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)	
     * @param max the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    CustomizedFS( double (*userFunction)(double) = NULL,
                  double min = -std::numeric_limits< double >::max(),
                  double max = std::numeric_limits< double >::max() ) : mep_userDefinedMF( userFunction ),
        m_minimum(min),
        m_maximum(max) {
    };

    /**
     * \brief Sets the membership function of the fuzzy set.
     * 
     * @param userFunction membership function defined by the user
     */
    void setMF( double (*userFunction)( double ) ){
        mep_userDefinedMF = userFunction;
    };

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( m_minimum );
    };

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
        return( m_maximum );
    }

protected:
    double mu(double x) const {
        return (*mep_userDefinedMF)(x);
    }

    double (*mep_userDefinedMF)( double );
    double m_minimum;
    double m_maximum;



};

}
#endif
