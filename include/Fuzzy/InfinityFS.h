
/**
 * \file InfinityFS.h
 *
 * \brief FuzzySet with a step function as membership function
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

#ifndef INFINITYFS_H
#define INFINITYFS_H
#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/FuzzySet.h>
#ifdef __SOLARIS__
#include <climits>
#endif
#ifdef __LINUX__
#include <float.h>
#endif

/**
 * \brief FuzzySet with a step function as membership function. 
 * 
 * The support of this FuzzySet reaches to either positive infinity or negative
 * infinity. The corresponding menbership function is discribed by two dedicated points a and b.
 * If the support reaches positive infinity, the membership function is 0 for values smaller than a,
 * raises constantly to 1 between a and b, and stays 1 for values greater than b. If the support 
 * reaches negantive infinity, the membership function is 1 for values smaller than a, falls 
 * constantly to 0 between a and b, and is 0 for values greater than b.  
 * 
 * For positive Infinity:
 * \f[
 *      \mu(x) = \left\{\begin{array}{ll} 0, & x < a \\ 
 *      \frac{1}{b-a}(x-a), & a \le x \le b \\ 
 *      1, & x > b\end{array}\right.
 * \f]
 * <img src="../images/InfinityFS.png"> 
 * 
 *
 *   
 */ //
class InfinityFS: virtual public FuzzySet {
public:
	// the following constructor allows for one side to be infinity,
	// p.ex. TFS1(true,0,1) yields
	//
	//           0 for x<=0
	// mu1(x) = { x for 0<x<=1
	//           1 for x>1
	//
	// IFS2(false,-1,0) yields mu2(x)=mu1(-x)


	
   /**
    * \brief Construnctor.
    * 
    * @param positiveInfinity decides whether the support reaches to positive (true) or
    * negative (false) infinity
    * @param a point where membership function starts to raise/fall (positiveInfinity: true/false)
    * @param b point where membership function stops to raise/fall (positiveInfinity: true/false)
    */
	InfinityFS(bool positiveInfinity,double a,double b);

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double  getMin() const{
		return(positiveInfinity?a:-std::numeric_limits<double>::max());
	};

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double  getMax() const{
		return(positiveInfinity?std::numeric_limits<double>::max():b);
	};

     /**
	* \brief Sets the parameters of the fuzzy set.
	*
     * @param positiveInfinity decides whether the support reaches to positive (true) or
     * negative infinity (false)
     * @param a point where membership function starts to raise/fall (positiveInfinity: true/false)
     * @param b point where membership function stops to raise/fall (positiveInfinity: true/false)
	*/
	void    setParams(bool positiveInfinity,double a,double b);
	
	
private:
	double         mu(double x) const;
	bool           positiveInfinity;
	double         a,b;
	
};

#endif
