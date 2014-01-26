/**
 *
 * \brief A composed LinguisticTerm
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef SHARK_FUZZY_COMPOSEDLT_H
#define SHARK_FUZZY_COMPOSEDLT_H

#include <shark/Fuzzy/LinguisticTerm.h>
#include <shark/Fuzzy/LinguisticVariable.h>

#include <shark/Fuzzy/FuzzySets/ComposedFS.h>

namespace shark {

/**
   * \brief A composed LinguisticTerm.
   *
   * A composed LinguisticTerm makes it possible to do some
   * calculations on fuzzy sets, e.g. to connect a constant fuzzy set 
   * and a sigmoid fuzzy set using the minimum fuction. This results
   * in a sigmoid fuzzy set which is cutted at the value of the 
   * constant fuzzy set.
   *
   */
class ComposedLT : public LinguisticTerm, public ComposedFS {
public:

    /**
     * \brief Constructor.
     * 
     * @param name the name
     * @param parent the associated linguistic variable 
     * @param op the operator used (MIN or MAX) to connect the fuzzy sets
     * @param f1 the first fuzzy set
     * @param f2 the second fuzzy set
     */
    ComposedLT(const std::string & name,
	       const boost::shared_ptr< LinguisticVariable > & parent,
	       Operator op,
	       const boost::shared_ptr<FuzzySet> & f1,
               const boost::shared_ptr<FuzzySet> & f2 ) : LinguisticTerm( name ),
        ComposedFS( op, f1, f2 ) {
        	setLinguisticVariable(parent);
    }
    

    /**
     * \brief Constructor.
     *
     * @param name the name
     * @param parent the associated linguistic variable 
     * @param op the operator used (MIN or MAX) to connect the fuzzy sets
     * @param f1 the first fuzzy set
     * @param f2 the second fuzzy set
     * @param userFunction the funcion connecting both fuzzy sets (a user definded operator 
     *        which is used instead of MIN or MAX) 
     */
    ComposedLT(const std::string & name = "ComposedLT",
               const boost::shared_ptr<LinguisticVariable> & parent = boost::shared_ptr<LinguisticVariable>(),
               Operator op = MIN,
               const boost::shared_ptr<FuzzySet> & f1 = boost::shared_ptr<FuzzySet>(),
               const boost::shared_ptr<FuzzySet> & f2 = boost::shared_ptr<FuzzySet>(),
               double (*userFunction)( double,double ) = NULL ) : LinguisticTerm( name ),
        ComposedFS( op, f1, f2, userFunction ) {
    }

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double min() const {
        return( std::max( ComposedFS::min(), m_parent->lowerBound() ) );
    };

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    inline double max() const {
        return( std::min( ComposedFS::max(), m_parent->upperBound() ) );
    }

};

ANNOUNCE_LINGUISTIC_TERM( ComposedLT, LinguisticTermFactory );

}
#endif
