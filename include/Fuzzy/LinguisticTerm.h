/**
 * \file LinguisticTerm.h
 *
 * \brief A single linguistic term
 * 
 * \author Marc Nunkesser
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



#ifndef LINGUISTICTERM_H
#define LINGUISTICTERM_H

#include <string> // new style header file ANSI C++
#include "FuzzySet.h"
#include "RCPtr.h"
#include "LinguisticVariable.h"

// this class is part of a diamond inheritance, thus we use virtual inheritance
class LinguisticVariable; // forward declaration


/**
 * \brief A single linguistic term.
 * 
 * A linguistic term (like 'fast' or 'slow') is a named FuzzySet, which is assigned to a
 * LinguisticVariable (like 'speed'). <br> 
 * This is a virtual base class. The children are different kinds of linguistic
 * terms corresponding to the different kinds of fuzzy sets. 
 */
class LinguisticTerm: virtual public FuzzySet {
public:

    /**
     * \brief Creates a new LinguisticTerm.
     * 
     * @param name name of the linguistic term, e.g. "very fast"
     */
	LinguisticTerm(const std::string name);
    
    /**
     * \brief Creates a new LinguisticTerm and assignes it to a given
     * LinguisticVariable.
     * 
     * @param name the name of the linguistic term, e.g. "very fast"
     * @param parent the assigned linguistic variable, e.g. "speed" 
     */
	LinguisticTerm(const std::string name="", const RCPtr<LinguisticVariable>& parent = NULL);
    
    /// Destructor
	virtual ~LinguisticTerm();

    /**
     * \brief Returns the name of the linguistic term.
     * @return the name
     */
	inline std::string getName() const {
		return(name);
	};
    
    /**
     * \brief Sets the name of the linguistic term.
     * @param name the new name
     */
	void setName(const std::string name);

    /**
     * \brief Returns the LinguisticVariable the term is assigned to.
     */
	inline const RCPtr<LinguisticVariable>&  getLinguisticVariable() const {
		return(parent);
	};
    
    /**
     * \brief Reassigns this term to a new LinguisticVariable.
     * @param lv the new linguistic variable
     */
	void setLinguisticVariable(const RCPtr<LinguisticVariable>& lv);


    /**
	* \brief Defuzzification of the linguistic term accounting the bounds of the suppport 
	*  given by the corresponding linguistic variable. 
  	*
     * @param errRel relative approximation error that is tollerated during numerical integration
     * @param recursionMax max. depth of recursion during numerical integration (i.e. max. \f$2^n\f$ steps)
	*/
	 double defuzzify( double errRel = ERR_RELATIVE, int recursionMax = RECURSION_MAX )const {
		return (FuzzySet::defuzzify(parent->getLowerBound(), parent->getUpperBound(), errRel, RECURSION_MAX));
	};




protected:
	RCPtr<LinguisticVariable>   parent; // Pointer to Linguistic Variable the term belongs to

private:
	std::string name;

};

#endif
