/**
 * \file ComposedFS.h
 *
 * \brief A composed FuzzySet
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

#ifndef COMPOSEDFS_H
#define COMPOSEDFS_H

#include <Fuzzy/FuzzySet.h>
#include <Fuzzy/RCPtr.h>
#include <Fuzzy/TriangularFS.h>

#include <algorithm>


class FuzzyException;

/**
 * \brief A composed FuzzySet.
 *
 * A composed FuzzySet makes it possible to do some calculations on fuzzy sets, 
 * e.g. to connect a constant fuzzy set and a sigmoid fuzzy set using the 
 * minimum fuction which would result in a sigmoid fuzzy set which is cutted at 
 * the value of the constant fuzzy set.
 */
class ComposedFS: virtual public FuzzySet {
public:
	enum Operator {
		MAX,
		MIN,
		PROD,
		PROBOR,
		USER,
		SIMPLIFY
	};

    /**
     * \brief Constructor.
     *
 	* @param op the operator used (e.g. MIN or MAX) to connect the fuzzy sets
 	* @param f1 the first fuzzy set
 	* @param f2 the second fuzzy set
 	*/
	ComposedFS(Operator op,
	           const RCPtr< FuzzySet >& f1,
	           const RCPtr< FuzzySet >& f2);

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
	           const RCPtr< FuzzySet >& f1,
	           const RCPtr< FuzzySet >& f2,
	           double (*userFunction) ( double, double ) );

    /**
	 * \brief Copy constructor.
     */
	inline ComposedFS(const ComposedFS&);


    /**
     * \brief Sets the operator.
  	 * 
 	 * @param o the operator to be used to connect two fuzzy sets
 	 */
	inline void setOperator(Operator o){
		op = o;
	};

    /**
	* \brief Returns the lower boundary of the support
	* 
	* @return the min. value for which the membership function is nonzero (or exceeds a
	* given threshold)
	*/
	double               getMin() const ;
    
    /**
	* \brief Returns the upper boundary of the support
	* 
	* @return the max. value for which the membership function is nonzero (or exceeds a
	* given threshold)
	*/
	double               getMax() const ;
	
private:

	Operator             op;
	double               mu(double) const;
	RCPtr< FuzzySet >    leftOperand;
	RCPtr< FuzzySet >    rightOperand;
	double         (*userDefinedOperator)(double,double); //Operator supplied by the user, which can be used instead of and/or
};

#endif
