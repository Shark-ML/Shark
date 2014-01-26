/**
*
* \brief Base class for n-dimensional fuzzy sets
* 
* \authors Marc Nunkesser
*/


#ifndef SHARK_FUZZY_NDIMFS_H
#define SHARK_FUZZY_NDIMFS_H

#include <shark/Fuzzy/FuzzySet.h>
#include <shark/Fuzzy/Rule.h>

#include <list>
#include <set>
#include <vector>
#include <algorithm>

namespace shark {

/**
 * \brief Base class for n-dimensional fuzzy sets.
 *
 * Virtual base class for the classes HomogenousNDimFS and ComposedNDimFS.
 * A n-dimensional fuzzy set is described by the corresponding membership
 * function:
 *
 * \f[
 *    \mu : X_1 \times X_2 \times \ldots \times X_n \rightarrow [0,1]
 * \f]
 * \f[
 *    \mu( x_1, x_2, \ldots, x_n ) = f( \mu_1(x_1), \mu_2(x_2),\ldots,\mu_n(x_n) )
 * \f]
 */
class NDimFS {
public:

    typedef std::list< boost::shared_ptr<FuzzySet> > FuzzyArrayType;

    /*enum Connective {
  AND,
  OR
  };*/

    /**
  * \brief Default Constructor.
  */
    NDimFS() {}

    /**
  * \brief Constructor.
  *
  * @param fat an array of fuzzy sets.
  */
    NDimFS( const FuzzyArrayType & fat ) : m_components( fat ) {
    }

    /**
  * \brief Constructor for conversion from class FuzzySet.
  *
  * @param f a fuzzy set.
  */
    NDimFS(const boost::shared_ptr< FuzzySet >& f ) {
        m_components.push_back( f );
    }

    virtual ~NDimFS() {
    }

    const FuzzyArrayType & components() const {
        return( m_components );
    }

    FuzzyArrayType & components() {
        return( m_components );
    }

    /**
  * \brief The n-dimensional \f$\mu\f$-function.
  *
  * @param v the vector of values \f$(x_1,\ldots,x_n)\f$
  * @return the value of the membership fuction at \f$(x_1,\ldots,x_n)\f$
  */
    virtual double operator()( const RealVector & v ) const = 0;

    /**
  * \brief Returns the dimension of a n-dimensional fuzzy set.
  *
  * @return the dimension of the fuzzy set.
  */
    inline std::size_t dimension() const{
        return( m_components.size() );
    };

protected:
    FuzzyArrayType m_components;
};

}

#endif
