/**
 * \file ConstantLT.h
 *
 * \brief Linguistic Term with constant membership function
 * 
 * \authors Asja Fischer and Björn Weghenkel
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


#ifndef COMPOSEDLT_H
#define COMPOSEDLT_H

#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/ConstantFS.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/RCPtr.h>

/**
 * \brief LinguisticTerm with constant membership function.
 * 
 * This class implements a LinguisticTerm with constant membership function.
 * 
 * <img src="../images/ConstantFS.png"> 
 * 
 */
class ConstantLT: public LinguisticTerm, public ConstantFS {
public:


    /**
	* \brief Constructor.
	*	
     * @param name the name
	* @param parent the associated linguistic variable 
	* @param x the constant value of the membership function
	*/
	ConstantLT(const std::string name,
	           const RCPtr<LinguisticVariable>&  parent,
			 double x); 

    /**
	* \brief Defuzzification of the linguistic variable.  
	*/
	inline double defuzzify() const{
		return( ( parent->getUpperBound() ) - ( parent->getLowerBound() ) / 2.0 );
	};


   /**
     * \brief Returns the lower boundary of the support
     * 
     * @return the min. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */

	inline double         getMin() const {
		return(std::max(ConstantFS::getMin(), parent->getLowerBound()));
	};

    /**
     * \brief Returns the upper boundary of the support
     * 
     * @return the max. value for which the membership function is nonzero (or exceeds a
     * given threshold)
     */	
	inline double         getMax() const {
		return(std::min(ConstantFS::getMax(), parent->getUpperBound()));
	};


	

};
#endif
