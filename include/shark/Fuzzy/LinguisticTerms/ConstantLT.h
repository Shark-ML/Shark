/**
 *
 * \brief Linguistic Term with constant membership function
 * 
 * \authors Asja Fischer and Björn Weghenkel
 */

/* $log$ */


#ifndef SHARK_FUZZY_CONSTANTLT_H
#define SHARK_FUZZY_CONSTANTLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/ConstantFS.h>

namespace shark {
/**
   * \brief LinguisticTerm with constant membership function.
   * 
   * This class implements a LinguisticTerm with constant membership function.
   * 
   * <img src="../images/ConstantFS.png"> 
   * 
   */
class ConstantLT : public LinguisticTerm, public ConstantFS {
public:
    /**
     * \brief Constructor.
     *	
     * @param name the name
     * @param parent the associated linguistic variable 
     * @param x the constant value of the membership function
     */
    ConstantLT(const std::string & name = "ConstantLT",
               const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
               double x = 1. ) : LinguisticTerm( name ),
        ConstantFS( x ) {
        setLinguisticVariable(parent);
    }

    /**
     * \brief Defuzzification of the linguistic variable.  
     */
    double defuzzify() const {
        return( ( m_parent->upperBound() ) - ( m_parent->lowerBound() ) / 2.0 );
    }

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( std::max( ConstantFS::min(), m_parent->lowerBound() ) );
    };

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
    double max() const {
        return( std::min( ConstantFS::max(), m_parent->upperBound() ) );
    }
};

}
#endif
