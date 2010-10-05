
/**
 * \file SingletonLT.h
 *
 * \brief LinguisticTerm with a single point of positive membership
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
#ifndef SINGLETONLT_H
#define SINGLETONLT_H

#include <Fuzzy/FuzzyException.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/SingletonFS.h>
#include <Fuzzy/LinguisticVariable.h>

/**
 * \brief LinguisticTerm with a single point of positive membership.
 * 
 * The membership function of a SingletonLT takes value 1 only at a given point
 * and value 0 everywhere else.
 */
class SingletonLT: public LinguisticTerm, public SingletonFS {
public:
	
    /**
	* \brief Constructor.
	* 
	* @param name the name
	* @param parent the associated linguistic variable   
	* @param p1 the value for which the membership function takes the value one
	*/
	SingletonLT(const std::string name,
	            const RCPtr<LinguisticVariable>& parent ,
	            double                     p1);


	// overloaded operator () - the mu function
	// inline double         operator()(double x) const;

    /**
     * \brief Returns the lower boundary of the support
	 * 
	 * @return the min. value for which the membership function is nonzero (or exceeds a
	 * given threshold)
	 */
	inline double         getMin() const {
		return(std::max(SingletonFS::getMin(), parent->getLowerBound()));
	};

    /**
 	 * \brief Returns the upper boundary of the support
 	 * 
	 * @return the max. value for which the membership function is nonzero (or exceeds a
	 * given threshold)
	 */
	inline double         getMax() const {
		return(std::min(SingletonFS::getMax(), parent->getUpperBound()));
	};

    /**
     * \brief Defuzzification
     *
     * If the support lies outside the range given by the associated linguistic
     * variable, the nearest bound will be taken for defuzzification.
     */
	double defuzzify() const;
  
};




///////////////////////////////////////////////
/////// inline functions
///////////////////////////////////////////////





//double SingletonLT::operator()(double x) const {return(SingletonFS::operator()(x));}


#endif
