
/**
 * \file FuzzyRelation.h
 *
 * \brief A fuzzy relation
 * 
 * \authors Marc Nunkesser, Copyright (c) 2008, Marc Nunkesser
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 */

/* $log$ */

#ifndef FUZZYRELATION_H
#define FUZZYRELATION_H

#include <vector>
#include "RCPtr.h"

class NDimFS;
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

	enum Lambda {X,Y};
	typedef double MuType(double, double);

	FuzzyRelation()  {};			///< Default Constructor 
	virtual ~FuzzyRelation() {};	///< Destructor

    /**
     * \brief Calculates the result of the relation between the two input vectos (implemented 
     * by subclass).
     * 
     * @param v1 the vector \f$(x_1,...,x_n)\f$ 
     * @param v2 the vector \f$(y_1,...,y_m)\f$ 
     * @return the value of the relation at \f$((x_1,...,x_n),(y_1,...,y_m))\f$
     */
	virtual double operator()(const std::vector<double> & v1, const std::vector<double> & v2) const = 0;
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
	virtual RCPtr<ComposedNDimFS> operator()( const std::vector<double> & v, Lambda lambda ) const = 0;
	
	/**
	 * \brief Calculates the relation with \f$x\f$ left as variable.
	 *     
	 * @param v the vector \f$(y_1,...y_m)\f$
	 * @param lambda the variable z 
     * @return the value of the relation at \f$(x,(y_1,...,y_m))\f$ 
	 */
	virtual RCPtr<ComposedNDimFS> operator()( Lambda lambda, const std::vector<double> & v ) const = 0;

private:

};

#endif

