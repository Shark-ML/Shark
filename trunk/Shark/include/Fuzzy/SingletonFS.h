/**
 * \file SingletonFS.h
 *
 * \brief FuzzySet with a single point of positive membership
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

#ifndef SINGLETONFS_H
#define SINGLETONFS_H

#include <Fuzzy/FuzzySet.h>
#include <cmath>

/**
 * \brief FuzzySet with a single point of positive membership.
 * 
 * The membership function of a SingletonFS takes value 1 only at a given point
 * and value 0 everywhere else.
 */
class SingletonFS: virtual public FuzzySet {
public:

    /**
	* \brief Constructor.
	* 
	* @param x position of membership point
	* @param yValue value of μ(x)
    * @param eps range of membership around x
	*/

	inline             SingletonFS(double x=0.0,
	                               double yValue = 1.0,
	                               double eps = 1E-5
	                              ):c(x), yValue(yValue),epsilon(eps) {};
	
    /**
     * \brief Defuzzification
     *
     */
	inline double      defuzzify() const{
		return c;
	};

   /**
    * \brief Returns the lower boundary of the support
	* 
	* @return the min. value for which the membership function is nonzero (or 
    * exceeds a given threshold)
	*/
	inline double      getMin() const {
		return c;
	};

    /**
 	* \brief Returns the upper boundary of the support
 	* 
	* @return the max. value for which the membership function is nonzero (or exceeds a
	* given threshold)
	*/
	inline double      getMax() const {
		return c;
	};

    /**
	* \brief Sets the parameters of the fuzzy set.
	* 
    * @param x new position of membership point
    * @param yValue new value of μ(x)
    * @param eps new range of membership around x
	*/
	inline void        setParams(double x=0.0,
	                             double yValue = 1.0,
	                             double eps = 1E-5);

private:
	inline double      mu( double x) const {
		return(fabs(x-c)<epsilon?yValue:0.0);
	};
	//the position at which the singleton is defined (mu(c)=value);
	double             c;
	//the precision to which the comparison is carried out

	double             yValue;
	double             epsilon;
};


#endif
