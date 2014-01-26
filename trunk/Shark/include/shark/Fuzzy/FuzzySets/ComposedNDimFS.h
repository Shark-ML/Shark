/**
*
* \brief A composed n-dimensional FuzzySet
* 
* \authors Marc Nunkesser
*/

/* $log$ */

#ifndef SHARK_FUZZY_COMPOSEDNDIMFS_H
#define SHARK_FUZZY_COMPOSEDNDIMFS_H

#include <shark/Fuzzy/FuzzySets/NDimFS.h>
#include <shark/Fuzzy/FuzzySets/ComposedFS.h>

namespace shark {
/**
 * \brief A composed n-dimensional FuzzySet.
 *
 * A composed n-dimensional fuzzy set makes it possible to do some
 * calculations on n-dimensopnal fuzzy sets,
 * e.g. to connect two n-dimensional fuzzy sets using the max-function.
 */
class ComposedNDimFS : public NDimFS {
public:

    /**
  * \brief Constructor.
  *
  *@param nDimFS1 the first n-dimensional fuzzy set
  *@param nDimFS2 the secound n-dimensional fuzzy set
  *@param userFunction the function connecting both fuzzy sets
  */
    ComposedNDimFS(const boost::shared_ptr<NDimFS> & nDimFS1,
                   const boost::shared_ptr<NDimFS> & nDimFS2,
                   double (*userFunction)( double, double ) ) : m_lhs( nDimFS1 ),
        m_rhs( nDimFS2 ),
        mep_userDefinedOperator( userFunction ) {
        if( m_lhs->dimension() != m_rhs->dimension() )
            throw( shark::Exception( "ComposedNDimFS: m_lhs->dimension() != m_rhs->dimension()", __FILE__, __LINE__ ) );
    }

    /**
  * \brief Membership (\f$\mu\f$) function.
  *
  * @param v the vector of values \f$(x_1,\ldots,x_n)\f$
  * @return the value of the membership fuction at \f$(x_1,\ldots,x_n)\f$
  */
    virtual double operator()( const RealVector & v ) const {
        return( mep_userDefinedOperator( (*m_lhs)( v ), (*m_rhs)( v ) ) );
    }


    /**
  * \brief Cast operator
  *
  * Casts the ComposedNDimFS to a ComposedFS if the dimension is equal to one.
  *
  * @return the ComposedFS
  */
    operator boost::shared_ptr<ComposedFS>() {
        return( boost::shared_ptr< ComposedFS >( new ComposedFS( ComposedFS::USER, m_lhs->components().front(), m_rhs->components().front() ) ) );
    }

    /**
  * \brief Returns the dimension of a n-dimensional fuzzy set.
  *
  * @return the dimension of the fuzzy set.
  */
    virtual std::size_t dimension() const {
        return( m_lhs->dimension() );
    };

protected:
    boost::shared_ptr<NDimFS> m_lhs;
    boost::shared_ptr<NDimFS> m_rhs;
    double (*mep_userDefinedOperator)( double, double );
};

}
#endif
