
/**
 *
 * \brief A LinguisticTerm with an user defined mambership function
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */
#ifndef SHARK_FUZZY_CUSTOMIZEDLT_H
#define SHARK_FUZZY_CUSTOMIZEDLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/CustomizedFS.h>

namespace shark {

/**
   * \brief A LinguisticTerm with an user defined mambership function.
   *
   * This class implements a LinguisticTerm with an user definded membership 
   * function.
   */
class CustomizedLT : public LinguisticTerm, public CustomizedFS {
public:

    /**
     * \brief Constructor.
     *
     * @param name the name
     * @param parent the associated linguistic variable 
     * @param userFunction membership function defined by the user
     * @param min the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)	
     * @param max the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    CustomizedLT( const std::string & name = "CustomizedLT",
                  const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
                  double (*userFunction)( double ) = NULL,
                  double min = -std::numeric_limits< double >::max(),
                  double max = std::numeric_limits< double >::max() ) : LinguisticTerm( name ),
        CustomizedFS( userFunction, min, max ) {
        setLinguisticVariable(parent);
    }

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const{
        return( std::max( CustomizedFS::min(), m_parent->lowerBound() ) );
    }

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const{
        return( std::min( CustomizedFS::max(), m_parent->upperBound() ) );
    }

};

}
#endif
