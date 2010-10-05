
/**
 * \file BellFS.h
 *
 * \brief FuzzySet with a bell-shaped (Gaussian) membership function
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
#ifndef BELLFS_H
#define BELLFS_H

#include <SharkDefs.h>
#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>

#include <climits>
#include <cassert>

/**
 * \brief FuzzySet with a bell-shaped (Gaussian) membership function.
 * 
 * This class implements a FuzzySet with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-offset)^2}{2\sigma^2}}
 * \f]
 * 
 * <img src="../images/BellFS.png">
 *  
 */
class BellFS: virtual public FuzzySet {
public:
	
    /**
 	* \brief Constructor.
	* 
	* @param sigma controlls the width of the Gaussian
	* @param offset position of the center of the peak
 	* @param scale scales the whole function
	*/
	BellFS( double sigma, double offset, double scale = 1 );


	// The bell mf is represented
	// by three parameters sigma, offset and scale:
	// bell(sigma,offset,scale) = 1/(sigma*sqrt(2pi))*exp(-(sigma*(x-offset))²/2sigma^2)
	//
    
    /**
     * \brief Defuzzifies the set by returning the Bell's offset.
     * 
     */
	virtual double defuzzify() const {
		return(offset);
	};

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	virtual double getMin() const {
		return(mn);
	};

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	virtual double getMax() const {
		return(mx);
	};

    /**
	* \brief Returns the theshold under which values of the function are set to be zero.
	* 
	* @return the theshold
	*/
	double getThreshold() const {
		return(threshold);
	};

	/**
	* \brief Sets the theshold under which values of the function are set to be zero.
	* 
	* @param thresh the theshold
	*/
	void setThreshold( double thresh );

	/**
 	* \brief Sets the parameters of the fuzzy set.
	* 
	* @param sigma controlls the width of the Gaussian
	* @param offset position of the center of the peak
 	* @param scale scales the whole function
	*/
	void setParams(double sigma, double offset, double scale = 1);

private:
	// overloaded operator () - the mu-function
	double                mu( double x ) const;

	double                sigma,offset,scale;// parameters of a sigmoidal MF
	double                threshold;
	double                mn,mx; // to avoid repeated calculation, these are stored.
	void                  setMaxMin(); // sets mn,mx
	static const double   factor; //  = 0.5*M_2_SQRTPI*M_SQRT1_2; //1/(sqrt(2pi))
	static const double   factor2; // = 2*M_SQRT2/M_2_SQRTPI; //sqrt(2pi)
};

#endif
