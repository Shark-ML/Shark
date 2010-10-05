
/**
 * \file Operators.h
 *
 * \brief Operators and connective functions
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


#ifndef OPERATORS_H
#define OPERATORS_H

#include <Fuzzy/FuzzySet.h>
#include <Fuzzy/SingletonFS.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/Implication.h>
#include <vector>
#include <Fuzzy/NDimFS.h>

/**
 * \brief Operators and connective functions.
 *
 * This class implements operators needed to do some calculations on fuzzy sets and the 
 * connective functions MINIMUM, MAXIMUM, PROD and PRODOR.	
 */
class Operators {
public:

/**
 * \brief Connects two fuzzy sets via the maximum function and returns the resulting fuzzy set.
 * 
 * @param f1 the first fuzzy set
 * @param f2 the second fuzzy set
 * @return the resulting composed fuzzy set
 */
	static RCPtr<ComposedFS>  max(const RCPtr<FuzzySet>& f1, const RCPtr<FuzzySet>& f2);


	// inline static RCPtr<SingletonFS> min(const RCPtr<FuzzySet>& ,const RCPtr<SingletonFS>&);
	// inline static RCPtr<SingletonFS>  min(const RCPtr<SingletonFS>&,const RCPtr<FuzzySet>& );

/**
 * \brief Connects two fuzzy sets via the minimum function and returns the resulting fuzzy set.
 * 
 * @param f1 the first fuzzy set
 * @param f2 the second fuzzy set
 * @return the resulting composed fuzzy set
 */
	static RCPtr<FuzzySet> min( const RCPtr<FuzzySet>& f1, const RCPtr<FuzzySet>& f2 );
	// sup min composition for (vector of ) singleton input:
	// instead of singletons one must pass directly their
	// double values i.e. Singleton.defuzzify()


/**
 * \brief Sup-min composition of an implication and a vector of singeltons.
 *
 * \f[
 *      \mu(y) = \sup_{x} min(\mu_1(x), \mu_2(x,y))
 * \f]
 * where \f$\mu_2(x,y)\f$ is the implication function
 * and \f$\mu_1(x)\f$ is the current input, in our case a 
 * vector of singletons, which simplifies this to
 *
 * \f[
 *      \mu(y) =\mu_2(singeltonInput, y)
 * \f]	
 *
 * @param input the input vector (vector of singeltons)
 * @param imp the implication
 * @return the resulting composed n-dimensional fuzzy set 
 */
	static RCPtr<ComposedNDimFS> supMinComp(const std::vector<double> & input, Implication* imp);

// Connective Functions

/**
 * \brief The minimum function.
 * 
 * @param a first input value
 * @param b second input value
 * retrun the minimum of a and b
 */
	inline static double minimum( double a, double b ) {
		return(b < a ? b : a);
	};

/**
 * \brief The maximum function.
 * 
 * @param a first input value
 * @param b second input value
 * retrun the maximum of a and b
 */
	inline static double maximum( double a, double b ) {
		return(a < b ? b : a);
	};

/**
 * \brief The PROD function.
 * 
 * @param a first input value
 * @param b second input value
 * retrun the product of a and b (a*b)
 */
	inline static double prod( double a, double b ) {
		return(a*b);
	};


/**
 * \brief The PROBOR function.
 * 
 * @param a first input value
 * @param b second input value
 * retrun PROBOR(a,b)= a+b-a*b
 */
	inline static double probor( double a, double b ) {
		return(a+b-a*b);
	};
	
private:
	
	inline static  RCPtr<SingletonFS> minLFS(const RCPtr<FuzzySet>&  s,const RCPtr<FuzzySet>& fs);
};

#endif
