
/**
 * \file BellLT.h
 *
 * \brief LinguisticTerm with a bell-shaped (Gaussian) membership function
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
#ifndef BELLLT_H
#define BELLLT_H

#include <Fuzzy/BellFS.h>
#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/LinguisticVariable.h>

#include <algorithm>
#include <string>

/**
 * \brief LinguisticTerm with a bell-shaped (Gaussian) membership function
 * 
 * This class implements a LinguisticTerm with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x-offset)^2}{2\sigma^2}}
 * \f]
 * 
 * <img src="../images/BellFS.png">
 *  
 */
class BellLT: public LinguisticTerm, public BellFS {
public:

    /**
 	* \brief Constructor.
	* 
	* @param name the name
	* @param parent the associated linguistic variable 
	* @param sigma controlls the width of the Gaussian
	* @param offset position of the center of the peak
 	* @param scale scales the whole function
	*/
	BellLT(const std::string                 name,
	       const RCPtr<LinguisticVariable>&  parent,
	       double sigma,
	       double offset,
	       double scale = 1);

     
    /**
     * \brief Defuzzifies the set by returning the Bell's offset if the support is not 
 	 *  bounded by the associated linguistic variable, and by dertermination of the center
	 *  of gravity otherwise. 
     * 
     * @param errRel relative approximation error that is tollerated during numerical integration
     * @param recursionMax max. depth of recursion during numerical integration (i.e. max. \f$2^n\f$ steps)
     */
     	double defuzzify(double errRel = ERR_RELATIVE, 
                           int recursionMax = RECURSION_MAX )const;

    /**
     * \brief Returns the lower boundary of the support.
     * 
     * @return the min. value for which the FuzzySet is nonzero (or exceeds a
     * given threshold)
     */
	inline double  getMin() const {
		return(std::max(BellFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support.
     * 
     * @return the max. value for which the FuzzySet is nonzero (or exceeds a
     * given threshold)
     */
	inline double getMax() const {
		return(std::min(BellFS::getMax(), parent->getUpperBound()));
	};
};

#endif
