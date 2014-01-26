
/**
 *
 * \brief LinguisticTerm with a single point of positive membership
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */
#ifndef SHARK_FUZZY_SINGLETONLT_H
#define SHARK_FUZZY_SINGLETONLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/SingletonFS.h>

namespace shark {
/**
   * \brief LinguisticTerm with a single point of positive membership.
   * 
   * The membership function of a SingletonLT takes value 1 only at a given point
   * and value 0 everywhere else.
   */
class SingletonLT : public LinguisticTerm, public SingletonFS {
public:

    /**
     * \brief Constructor.
     * 
     * @param name the name
     * @param parent the associated linguistic variable   
     * @param p1 the value for which the membership function takes the value one
     */
    SingletonLT(const std::string & name = "SingletonLT",
                const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
                double p1 = 0.) : LinguisticTerm( name ),
        SingletonFS( p1, 1 ) {
        setLinguisticVariable(parent);
    }

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
        return( std::max( SingletonFS::min(), m_parent->lowerBound() ) );
    }

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
        return( std::min( SingletonFS::max(), m_parent->upperBound() ) );
    }

    /**
     * \brief Defuzzification
     *
     * If the support lies outside the range given by the associated linguistic
     * variable, the nearest bound will be taken for defuzzification.
     */
    double defuzzify() const {
        double result = SingletonFS::defuzzify();
        result = std::min( result, m_parent->upperBound() );
        result = std::max( result, m_parent->lowerBound() );
        return result;
    }

};
ANNOUNCE_LINGUISTIC_TERM( SingletonLT, LinguisticTermFactory );
}
#endif
