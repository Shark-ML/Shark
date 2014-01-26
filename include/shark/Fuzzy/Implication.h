
/**
 *
 * \brief Fuzzy implication.
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */
#ifndef SHARK_FUZZY_IMPLICATION_H
#define SHARK_FUZZY_IMPLICATION_H

#include <shark/Fuzzy/FuzzySet.h>
#include <shark/Fuzzy/FuzzyRelation.h>

#include <shark/Fuzzy/FuzzySets/ComposedNDimFS.h>
#include <shark/Fuzzy/FuzzySets/ConstantFS.h>
#include <shark/Fuzzy/FuzzySets/HomogenousNDimFS.h>

#include <boost/shared_ptr.hpp>

#include <algorithm>

namespace shark {

  class NDimFS;

  /**
   * \brief Fuzzy implication.
   *
   * This class enables the user to configurate an implication by his own.
   *
   * <table border=1>
   *   <tr>
   *     <td>Zadeh</td>
   *     <td>\f$I_{Zad}(x,y)=\max(\min(x,y),1-x)\f$</td>
   *     <td>ZADEH</td>
   *   </tr>
   *   <tr>
   *     <td>Mamdani (Minimum)</td>
   *     <td>\f$I_{Mam}(x,y)=\min(x,y)\f$</td>
   *     <td>MAMDANI</td>
   *   </tr>
   *   <tr>
   *     <td>Lukasiewicz</td>
   *     <td>\f$I_{Luk}(x,y)=\min(1,1-x+y)\f$</td>
   *     <td>LUKASIEWICZ</td>
   *   </tr>
   *   <tr>
   *     <td>Goedel (Standard Star)</td>
   *     <td>\f$I_{Goed}(x,y)=\left\{\begin{array}{ll} 1 & \mbox{for } x \leq y \\ 
   *      y & \mbox{otherwise} \end{array}\right.\f$</td>
   *     <td>GOEDEL</td>
   *   </tr>
   *   <tr>
   *     <td>Kleene-Dienes</td>
   *     <td>\f$I_{Kle}(x,y)=\max(1-x,y)\f$</td>
   *     <td>KLEENEDIENES</td>
   *   </tr>
   *   <tr>
   *     <td>Goguen (Gaines)</td>
   *     <td>\f$I_{Gog}(x,y)=\left\{\begin{array}{ll} 1 & \mbox{for } x=0\\
   *     \min(1,y/x) & \mbox{otherwise}\end{array}\right.\f$</td>
   *     <td>GOGUEN</td>
   *   </tr>
   *   <tr>
   *     <td>Gaines-Reschner (Standard Strict)</td>
   *     <td>\f$I_{Gai}(x,y)=\left\{\begin{array}{ll} 1 & \mbox{for } x \leq y \\
   *     0 & \mbox{otherwise} \end{array}\right.\f$</td>
   *     <td>GAINESRESCHER</td>
   *   </tr>
   *   <tr>
   *     <td>Reichenbach(algebraic implication)</td>
   *     <td>\f$I_{Rei}(x,y)=1-x+xy\f$</td>
   *     <td>REICHENBACH</td>
   *   </tr>
   *   <tr>
   *     <td>Larsen</td>
   *     <td>\f$I_{Lar}(x,y)=xy\f$</td>
   *     <td>LARSEN</td>
   *   </tr>
   * </table> 
   * 
   */
  class Implication : public FuzzyRelation {
  public:
    enum ImplicationType {
      ZADEH,
      MAMDANI,
      LUKASIEWICZ,
      GOEDEL,
      KLEENEDIENES,
      GOGUEN,
      GAINESRESCHER,
      REICHENBACH,
      LARSEN
    };

    /**
     * \brief Constructor.
     *
     * @param NDim1 first n-dimensional fuzzy set
     * @param NDim2 second n-dimensional fuzzy set
     * @param it the type of implication
     */ 
    Implication( const boost::shared_ptr<NDimFS>& NDim1,
		 const boost::shared_ptr<NDimFS>& NDim2,
		 ImplicationType it
		 ) : m_xfs( NDim1 ),
      m_yfs( NDim2 ) {

	switch( it ) {
	  case ZADEH:
		mep_function = Zadeh;
		break;
	case MAMDANI:
		mep_function = Mamdani;
		break;
	case LUKASIEWICZ:
		mep_function = Lukasiewicz;
		break;
	case GOEDEL:
		mep_function = Goedel;
		break;
	case KLEENEDIENES:
		mep_function = KleeneDienes;
		break;
	case GOGUEN:
		mep_function = Goguen;
		break;
	case GAINESRESCHER:
		mep_function = GainesRescher;
		break;
	case REICHENBACH:
		mep_function = Reichenbach;
		break;
	case LARSEN:
		mep_function = Larsen;
		break;
	}
      }

    /**
     * \brief Destructor.
     */
    virtual ~Implication() {}

    // overloaded operator():
	
    /**
     * \brief Calculates the value of the implication \f$R(x,y)\f$ for the given 
     * points.
     * 
     * @param x \f$x\f$
     * @param y \f$y\f$
     */
    virtual double operator()(const RealVector & x,const RealVector & y) const {
      if( mep_function == NULL )
	return( -1. );

      return( mep_function( (*m_xfs)( x ), (*m_yfs)( y ) ) );      
    }
	
    /**
     * \brief Calculates the implication given \f$x\f$ and \f$y\f$ left 
     * variable.
     * 
     * @param x \f$x\f$
     * @param lambda use: Lambda::Y
     * 
     * @return the implication
     */
    virtual boost::shared_ptr<ComposedNDimFS> operator()(const RealVector & x, Lambda lambda) const {
      if( lambda != Y )
	throw( shark::Exception( "Lambda::Y expected", __FILE__, __LINE__ ) );

      boost::shared_ptr< ConstantFS > constFS( new ConstantFS( (*m_xfs)( x ) ) );
      std::list< boost::shared_ptr< FuzzySet > > l( 1, constFS );
      
      boost::shared_ptr< HomogenousNDimFS > ndfs( new HomogenousNDimFS( l ) );
      boost::shared_ptr< ComposedNDimFS > compFS( new ComposedNDimFS( ndfs, m_yfs, mep_function ) );

      return( compFS );
    }
	
    /**
     * \brief Returns 0.
     */
    virtual boost::shared_ptr<ComposedNDimFS> operator()(Lambda , const RealVector & ) const {
      return( boost::shared_ptr<ComposedNDimFS>() );
    }

    static double Zadeh( double x, double y ) {
      return( std::max( std::min( x, y ), 1 - x ) );
    }
    
    static double Mamdani( double x, double y ) {
      return( std::min( x, y ) );
    }

    static double Lukasiewicz( double x, double y ) {
      return( std::min( 1., 1. - x + y ) );
    }

    static double Goedel( double x, double y ) {
      return( x <= y ? 1 : y );
    }

    static double KleeneDienes( double x, double y ) {
      return( std::max( 1 - x, y ) );
    }

    static double Goguen( double x, double y) {
      return( ::fabs( x ) < 1E-8 ? 1. : std::min( 1., y/x ) ); 
    }

    static double GainesRescher( double x, double y ) {
      return( x <= y ? 1. : 0. );
    }

    static double Reichenbach( double x, double y ) {
      return( 1. - x * x*y );
    }

    static double Larsen( double x, double y ) {
      return( x * y );
    }

  protected:
    typedef double (*Function)( double, double );
    Function mep_function;

    const boost::shared_ptr<NDimFS> m_xfs;
    const boost::shared_ptr<NDimFS> m_yfs;
  };
}

#endif
