/**
 * \file GeneralizedBellLT.h
 *
 * \brief LinguisticTerm with a generalized bell-shaped membership function.
 * 
 * \authors Thomas Voß
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



#ifndef __GENERALIZEDBELLLT_H__
#define __GENERALIZEDBELLLT_H__

#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/GeneralizedBellFS.h>

/**
 * \brief LinguisticTerm with a generalized bell-shaped membership function.
 * 
 * This class implements a LinguisticTerm with membership function:
 * 
 * \f[
 *      \mu(x) = \frac{1}{1+ (\frac{x-center}{width})^{2*slope}}
 * \f]
 * <img src="../images/GeneralizedBellFS.png"> 
 */
class GeneralizedBellLT : public LinguisticTerm, public GeneralizedBellFS {
public:

/**
 * \brief Constructor.
 *
 * @param name the name of the LinguisticTerm
 * @param parent the associated LinguisticVariable
 * @param slope the slope
 * @param center the center of the bell-shaped function
 * @param width the width of the bell-shaped function
 * @param scale the scale of the bell-shaped function
 */
	GeneralizedBellLT( const std::string & name,
					   const RCPtr<LinguisticVariable> & parent,
					   double slope = 1.0,
					   double center = 0.0,
					   double width = 1.0,
					   double scale = 1.0
					   );


   /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */
	inline double         getMin() const {
		return(std::max(GeneralizedBellFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
	inline double         getMax() const {
		return(std::min(GeneralizedBellFS::getMax(), parent->getUpperBound()));
	};


};

#endif // __GENERALIZEDBELLLT_H__

