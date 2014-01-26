
/**
 *
 * \brief A fuzzy relation
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */

#ifndef SHARK_FUZZY_FUZZYRELATION_H
#define SHARK_FUZZY_FUZZYRELATION_H

#include <shark/LinAlg/Base.h>

#include <boost/shared_ptr.hpp>

#include <vector>

namespace shark {

  class ComposedFS;
  class ComposedNDimFS;

  /**
   * \brief A fuzzy relation.
   *
   * A virtual basis class. An Implication is a speciall kind of a fuzzy
   * relation.
   */
  class FuzzyRelation {
  public:
    enum Lambda {
      X,
      Y
    };
    
    typedef double MuType(double, double);

    virtual ~FuzzyRelation() {}

    /**
     * \brief Calculates the result of the relation between the two input vectos (implemented 
     * by subclass).
     * 
     * @param v1 the vector \f$(x_1,...,x_n)\f$ 
     * @param v2 the vector \f$(y_1,...,y_m)\f$ 
     * @return the value of the relation at \f$((x_1,...,x_n),(y_1,...,y_m))\f$
     */
    virtual double operator()(const RealVector & v1, const RealVector & v2) const = 0;
    // A kind of Lambda calculus is possible with Fuzzy Relations.
    // For example letting x be 5.0 and leaving y a variable is
    // myFuzzyRelation(5.0,Lambda::Y)
    // Letting y be 5.0 and x a variable is myFuzzyRelation(Lambda::X,5.0)
    // myFuzzyRelation(5.0,Lambda::X) is undefined as well as
    // myFuzzyRelation(Lambda::X,Lambda::y)
	
    /**
     * \brief Calculates the relation with \f$y\f$ left as variable.
     *     
     * @param v the vector \f$(x_1,...,x_n)\f$
     * @param lambda the variable y 
     * @return the value of the relation at \f$((x_1,...,x_n),y)\f$ 
     */
    virtual boost::shared_ptr<ComposedNDimFS> operator()( const RealVector & v, Lambda lambda ) const = 0;
	
    /**
     * \brief Calculates the relation with \f$x\f$ left as variable.
     *     
     * @param v the vector \f$(y_1,...y_m)\f$
     * @param lambda the variable z 
     * @return the value of the relation at \f$(x,(y_1,...,y_m))\f$ 
     */
    virtual boost::shared_ptr<ComposedNDimFS> operator()( Lambda lambda, const RealVector & v ) const = 0;

  };
}
#endif

