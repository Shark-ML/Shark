
/**
 * \file SigmoidalLT.h
 *
 * \brief LinguisticTerm with sigmoidal membership function
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

#ifndef SIGMOIDALLT_H
#define SIGMOIDALLT_H

// #include <ZNminmax.h>

#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/SigmoidalFS.h>

/**
 * \brief LinguisticTerm with sigmoidal membership function.
 * 
 * This class implements a LinguisticTerm with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{1 + e^{-a(x-b)}}
 * \f]
 * 
 * <img src="../images/SigmoidalFS.png"> 
 * 
 */
class SigmoidalLT: public LinguisticTerm, public SigmoidalFS {
public:
	
	//            SigmoidalLT(double paramC,double paramOffset);

    /**
	* \brief Constructor.
	*
	* @param name the name
	* @param parent the associated linguistic variable
    * @param paramC scale factor for y-axis
    * @param paramOffset position of inflection point
	*/	
	SigmoidalLT(const std::string name,
	            const RCPtr<LinguisticVariable>& parent,
	            double                           paramC = 1,
	            double                           paramOffset = 0);

    /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMin() const {
		return(std::max(SigmoidalFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMax() const {
		return(std::min(SigmoidalFS::getMax(), parent->getUpperBound()));
	};

};


#endif
