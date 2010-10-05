
/**
 * \file HomogenousNDimFS.h
 *
 * \brief A homogenous n-dimensional fuzzy set
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


#ifndef HOMOGENOUSNDIMFS_H
#define HOMOGENOUSNDIMFS_H

#include <Fuzzy/FuzzySet.h>
#include <vector>
#include <algorithm>
#include <Fuzzy/Rule.h>
#include <Fuzzy/NDimFS.h>
#include <Fuzzy/RCPtr.h>


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
class HomogenousNDimFS:public NDimFS {
public:

	typedef std::vector< RCPtr<FuzzySet> > FuzzyArrayType;
	//enum                      Connective {AND, OR}; //order is important

/**
 * \brief Constructor.
 *
 * @param fat an array of fuzzy sets.
 * @param con the connective function (AND, OR)
 */
	HomogenousNDimFS( FuzzyArrayType & fat, Connective con = AND);

/**
 * \brief Constructor for conversion from class FuzzySet.
 *
 * @param f a fuzzy set
 */
	HomogenousNDimFS(const RCPtr<FuzzySet>& f);
	
/** 
 * \brief NDimFS destructor.
 */
	//virtual                   ~NDimFS();

/** 
 * \brief Destructor.
 */
	virtual ~HomogenousNDimFS();

/**
 * \brief Membership (\f$\mu\f$) function.
 * 
 * @param v the vector of values \f$(x_1,\ldots,x_n)\f$
 * @return the value of the membership fuction at \f$(x_1,\ldots,x_n)\f$
 */
    double operator()(const std::vector<double> & v) const;
	
/**
 * \brief Membership (\f$\mu\f$) function for one-dimensional fuzzy set.
 * 
 * @param a the value \f$x\f$
 * @return the value of the membership fuction at \f$x\f$
 */	
	double operator()(double a) const;
	
/**
 * \brief Membership (\f$\mu\f$) function for two-dimensional fuzzy set.
 * 
 * @param a the value \f$x_1\f$
 * @param b the value \f$x_2\f$
 * @return the value of the membership fuction at \f$(x_1,x_2)\f$
 */	
	double operator()(double a, double b) const;
	
/**
 * \brief Membership (\f$\mu\f$) function for three-dimensional fuzzy set.
 * 
 * @param a the value \f$x_1\f$
 * @param b the value \f$x_2\f$
 * @param c the value \f$x_3\f$
 * @return the value of the membership fuction at \f$(x_1,x_2,x_3)\f$
 */		
	double operator()(double a, double b, double c) const;
	
/**
 * \brief Membership (\f$\mu\f$) function for four-dimensional fuzzy set.
 * 
 * @param a the value \f$x_1\f$
 * @param b the value \f$x_2\f$
 * @param c the value \f$x_3\f$
 * @param d the value \f$x_4\f$
 * @return the value of the membership fuction at \f$(x_1,x_2,x_3,x_4)\f$
 */		
	double operator()(double a,double b, double c, double d) const;
	//
	// double              operator()(...) const;
private:
	Connective                compoConnective;
	typedef double ConnectiveFuncType(double, double);
	ConnectiveFuncType*       connectiveFunc;

	void                      setConnective(Connective);
};

#endif
