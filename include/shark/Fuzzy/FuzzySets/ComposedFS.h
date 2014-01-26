/**
 *
 * \brief A composed FuzzySet
 * 
 * \authors Marc Nunkesser
 */
/* $log$ */

#ifndef COMPOSEDFS_H
#define COMPOSEDFS_H

#include <shark/Fuzzy/FuzzySet.h>

#include <boost/shared_ptr.hpp>

#include <algorithm>

namespace shark {
  class FuzzyException;

  /**
   * \brief A composed FuzzySet.
   *
   * A composed FuzzySet makes it possible to do some calculations on fuzzy sets, 
   * e.g. to connect a constant fuzzy set and a sigmoid fuzzy set using the 
   * minimum fuction which would result in a sigmoid fuzzy set which is cutted at 
   * the value of the constant fuzzy set.
   */
  class ComposedFS : virtual public FuzzySet {
  public:
    enum Operator {
      MAX,
      MIN,
      PROD,
      PROBOR,
      USER
      /* SIMPLIFY */
    };

    /**
     * \brief Constructor.
     *
     * @param op the operator used (e.g. MIN or MAX) to connect the fuzzy sets
     * @param f1 the first fuzzy set
     * @param f2 the second fuzzy set
     */
    ComposedFS(Operator op,
	       const boost::shared_ptr< FuzzySet >& f1,
	       const boost::shared_ptr< FuzzySet >& f2
	       ) : m_operatorType( op ),
      m_lhs( f1 ),
      m_rhs( f2 ),
      mep_userDefinedOperator( NULL ) {
      }

    /**
     * \brief Constructor.
     *
     * @param op the operator used (e.g. MIN or MAX) to connect the fuzzy sets
     * @param f1 the first fuzzy set
     * @param f2 the second fuzzy set
     * @param userFunction the funcion connecting both fuzzy sets (a user 
     *        definded operator which is used instead of one of the stanard 
     *        operators) 
     */
    ComposedFS(Operator op,
	       const boost::shared_ptr< FuzzySet >& f1,
	       const boost::shared_ptr< FuzzySet >& f2,
	       double (*userFunction) ( double, double ) 
	       ) : m_operatorType( op ),
      m_lhs( f1 ),
      m_rhs( f2 ),
      mep_userDefinedOperator( userFunction ) {
    }

    /**
     * \brief Sets the operator.
     * 
     * @param o the operator to be used to connect two fuzzy sets
     */
    inline void setOperatorType( Operator o ) {
      m_operatorType = o;
    };

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double min() const {
      double l,r;
      switch( m_operatorType )  {
      case MIN:
	return(std::max(m_lhs->min(),m_rhs->min()));
	break;
      case MAX:
	return(std::min(m_lhs->min(),m_rhs->min()));
	break;
      case PROD:
	return(std::max(m_lhs->min(),m_rhs->min()));
	break;
      case PROBOR:
	l = m_lhs->min();
	r = m_rhs->min();
	return(std::min(l,r));
	break;
      case USER:
	return(std::min(m_lhs->min(),m_rhs->max()));
	// this is a simplification wich could lead to (slightly) incorrect
	// results depending on the user-defined operator
	break;
      default:
	throw(shark::Exception("Unknown member function type/operator", __FILE__, __LINE__ ) ); 
      }

      return( 0 );
    }
    
    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
    double max() const {
      double l,r;
      switch( m_operatorType )  {
      case MIN:
	l = m_lhs->max();
	r = m_rhs->max();
	// cout<<"Linker Operand:"<<l<<endl;
	// cout<<"Rechter Operand:"<<r<<endl;
	return( std::min( l,r) );
	break;
      case MAX:
	l=m_lhs->max();
	r=m_rhs->max();
	// cout<<"Linker Operand:"<<l<<endl;
	// cout<<"Rechter Operand:"<<r<<endl;
	return( std::max( l,r ) );
	break;
      case PROD:
	return( std::min( m_lhs->max(),m_rhs->max() ) );
	break;
      case PROBOR:
	l = m_lhs->max();
	r = m_rhs->max();
	return( std::max( l,r ) );
	break;
      case USER:
	l = m_lhs->max();
	r = m_rhs->max();
	return(std::max(l,r));
	break;
      default:
	throw( shark::Exception( "Unknown member function type/operator", __FILE__, __LINE__ ) );
      };
      return( 0 );
    }
    
  protected:

    double mu( double x ) const {
      double l,r;
      switch( m_operatorType ) {
      case MIN:
	return(std::min((*m_lhs)(x),(*m_rhs)(x)));
	break;
      case MAX:
	return( std::max( (*m_lhs)( x ),(*m_rhs)( x ) ) );
	break;
      case PROD:
	return( (*m_lhs)( x ) * (*m_rhs)( x ) );
	break;
      case PROBOR:
	l = (*m_lhs)( x );
	r = (*m_rhs)( x );
	return(l+r-l*r);
	break;
      case USER:
	return( (*mep_userDefinedOperator)( (*m_lhs)( x ),(*m_rhs)( x ) ) );
	break;
      default:
	throw( shark::Exception( "Unknown member function type/operator", __FILE__, __LINE__ ) );
      };
    }

    Operator m_operatorType;
    boost::shared_ptr< FuzzySet >    m_lhs;
    boost::shared_ptr< FuzzySet >    m_rhs;
    double (*mep_userDefinedOperator)(double,double); //Operator supplied by the user, which can be used instead of and/or
  };
}
#endif
