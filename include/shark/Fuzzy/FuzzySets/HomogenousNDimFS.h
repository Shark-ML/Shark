
/**
*
* \brief A homogenous n-dimensional fuzzy set
* 
* \authors Marc Nunkesser
*/

#ifndef SHARK_FUZZY_HOMOGENOUSNDIMFS_H
#define SHARK_FUZZY_HOMOGENOUSNDIMFS_H

#include <shark/Fuzzy/Fuzzy.h>

#include <shark/Fuzzy/FuzzySet.h>
#include <shark/Fuzzy/Operators.h>
#include <shark/Fuzzy/FuzzySets/NDimFS.h>

#include <list>
#include <set>
#include <vector>
#include <algorithm>

namespace shark {

/**
 *\brief A homogenous n-dimensional fuzzy set.
 *
 * A n-demensional fuzzy set is described by the corresponding membership
 * function:
 * \f[
 *    \mu : X_1 \times X_2 \times \ldots \times X_n \rightarrow [0,1]
 * \f]
 * \f[
 *    \mu( x_1, x_2, \ldots, x_n ) = f( \mu_1(x_1), \mu_2(x_2),\ldots,\mu_n(x_n) )
 * \f]
 *
 * In a homogenous n-dimensional fuzzy set the connective function f is the max-function
 * or the min-fuction over all µi.
 */
class HomogenousNDimFS : public NDimFS {
public:

    /// \brief Unary predicate taking a shared pointer to Fuzzy set and return it indicating whether the pointer refers to a non-empty component or not.
    struct NonEmptyComponent {
        bool operator()( const boost::shared_ptr< FuzzySet > & fs ) const {
            return( fs );
        }
    };

    typedef std::list< boost::shared_ptr< FuzzySet > > FuzzyArrayType;

    /**
  * \brief Constructor.
  *
  * @param fat an array of fuzzy sets.
  * @param con the connective function (AND, OR)
  */
    HomogenousNDimFS( const FuzzyArrayType & fat = FuzzyArrayType(), Connective con = AND ) : NDimFS( fat ) {
        setConnective( con );
    }

    /**
  * \brief Constructor for conversion from class FuzzySet.
  *
  * @param f a fuzzy set
  */
    HomogenousNDimFS( const boost::shared_ptr<FuzzySet> & f ) : NDimFS( f ) {
        setConnective( AND );
    }

    /**
  * \brief Destructor.
  */
    virtual ~HomogenousNDimFS() {}

    /**
  * \brief Membership (\f$\mu\f$) function.
  *
  * @param v the vector of values \f$(x_1,\ldots,x_n)\f$
  * @return the value of the membership fuction at \f$(x_1,\ldots,x_n)\f$
  */
    double operator()( const RealVector & v ) const {

        FuzzyArrayType::const_iterator it = std::find_if( m_components.begin(), m_components.end(), NonEmptyComponent() );

        if( it == m_components.end() )
            throw( shark::Exception( "Homogenous Fuzzy Set has no non-empty components.", __FILE__, __LINE__ ) );

        double result = (**it)( v( std::distance( m_components.begin(), it ) ) );

        RealVector::const_iterator iti = v.begin() + std::distance( m_components.begin(), it ) + 1;
        for( ++it; iti != v.end(); ++it, ++iti ) {
            if( *it == NULL )
                continue;

            result = mep_connectiveFunc( result, (**it)( *iti ) );
        }

        return( result );
    }

protected:
    void setConnective( Connective c ) {
        /*	m_compoConnective = c;
   switch( m_compoConnective ) {
   case AND :
    mep_connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::minimum);
    break;
   case OR :
    mep_connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::maximum);
    break;
   case PROD:
    mep_connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::prod);
    break;
   case PROBOR:
    mep_connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::probor);
    break;
   }*/
    }

    Connective m_compoConnective;
    typedef double ConnectiveFuncType(double, double);
    ConnectiveFuncType * mep_connectiveFunc;
};

}
#endif
