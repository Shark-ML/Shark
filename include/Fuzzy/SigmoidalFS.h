
/**
 * \file SigmoidalFS.h
 *
 * \brief FuzzySet with sigmoidal membership function
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
#ifndef SIGMOIDALFS_H
#define SIGMOIDALFS_H

#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>
#include <climits>
#include <cmath>
#include <cassert>

/**
 * \brief FuzzySet with sigmoidal membership function.
 * 
 * This class implements a FuzzySet with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{1 + e^{-a(x-b)}}
 * \f]
 * 
 * <img src="../images/SigmoidalFS.png"> 
 * 
 */
class SigmoidalFS: virtual public FuzzySet {
public:
	
    /**
	* \brief Constructor.
	*
	* @param paramC scale factor for y-axis
	* @param paramOffset position of inflection point
	*/	
	inline SigmoidalFS(double paramC = 1,double paramOffset = 0): c(paramC), offset(paramOffset), threshold(1E-6) {};

	// overloaded operator () - the mu-function
	// The sigmoidal mf (normalized i.e the max is always 1) is represented
	// by two parameters c and offset:
	// sig(c,offset) = 1/(1+exp(-c(x-offset))
	//

     /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double getMin() const {
		return(c<=0?(-std::numeric_limits<double>::min()):((-1/c)*log(1/threshold-1)+offset));
	};
	// Calculate where sigmoid becomes greater than the threshold value
	// by setting sig = threshold
	// which yields x=(-1/c)*ln(1-1/threshold)+offset)
	
	
	
	
     /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double getMax() const {
		return(c>=0?(std::numeric_limits<double>::max()):((-1/c)*log(1/threshold-1)+offset));
	} ;

    /**
 	* \brief Returns the threshold for which the membership function (for which the x axis
	* is the asymptote) is set to be zero in the numerical intergration.
	*
	* @return the threshold
	*/
	inline double getThreshold() const {
		return(threshold);
	};

    /**
 	* \brief Sets the threshold for which the membership function (for which the x axis
	* is the asymptote) is set to be zero in the numerical intergration.
	*
	* @param thr the threshold 
	*/
	void setThreshold(double thr);

    /**
	* \brief Sets the parameters of the fuzzy set.
	*
	* @param paramC 
	* @param paramOffset 
	*/	
	inline void setParams(double paramC,double paramOffset);

private:
	double  mu(double x) const;
	double  c,offset;// parameters of a sigmoidal MF
	double  threshold;
};




#endif
