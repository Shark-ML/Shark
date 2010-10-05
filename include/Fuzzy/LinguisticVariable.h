/**
 * \file LinguisticVariable.h
 *
 * \brief A composite of linguistic terms
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

#ifndef LINGUISTICVARIABLE_H
#define LINGUISTICVARIABLE_H

#include "FuzzyException.h"
#include <string>
#ifdef __SOLARIS__
#include <climits>
#endif
#ifdef __LINUX__
#include <float.h>
#endif
#include <map>
#include <limits>
#include <list>
#include "RCPtr.h"
#include "RCObject.h"

class LinguisticTerm; // forward declaration

/**
 * \brief A composite of linguistic terms.
 * 
 * A linguistic variable combines linguistic terms like 'slow', 'moderately fast',
 * 'fast', 'unbelievable fast' under a new name, e.g. 'speed'.  
 */
class LinguisticVariable: virtual public RCObject {

public:
	/**
	 * \brief The constructor sets the name and the range for the new linguistic 
	 * variable, e.g. ( "Temperature", -10, 120 ).
     * 
	 * The bounds also hold for the support of all derived fuzzy sets, i.e.
     * regions outside these bounds are ignored for defuzzyfication.
     *  
	 * @param name the variables name, e.g. "Temperature"
	 * @param lowerBound lower bound of the variables range
	 * @param upperBound upper bound of the variables range
	 */
	LinguisticVariable(
	    const std::string name = "",
	    double lowerBound = -std::numeric_limits<double>::max(),
	    double upperBound = std::numeric_limits<double>::max());

	/// Destructor
	virtual                    ~LinguisticVariable();

    /// Returns the name of the linguistic variable.
	std::string                getName() const;
	
	/**
	 * \brief Sets the name of the linguistic variable.
	 * @param name the new name
	 */
	void                       setName(const std::string name);
	
	/**
	 * \brief Adds a new linguistic term.
     * 
     * This method shouldn't be necessary most of time, since a LinguisticTerm
     * is associated with its LinguisticVariable durin construction.
     * 
	 * @param lt the LinguisticTerm to add
	 */
	void                       addTerm(LinguisticTerm * lt);
    
    /**
     * \brief Removes the given LinguisticTerm from this variable.
     * @param lt the LinguisticTerm to remove
     */
	void                       removeTerm(LinguisticTerm * lt);
    
    /// Returns number of terms associated with the LinguisticVariable
	inline int                 getNumberOfTerms() {
		return(terms.size());
	};
    
    /**
     * \brief Returns a LinguisticTerm associated with this variable.
     * @param whichOne index of the term ( 0 <= index < numerOfTerms )
     */
	const RCPtr<LinguisticTerm> getTerm(int whichOne);
    
    /**
     * \brief Sets a new range for the variable.
     *  
     * The bounds also hold for the support of all derived fuzzy sets.
     *  
     * @param lower the lower bound
     * @param upper the upper bound
     */
	void setBounds(double lower, double upper);
    
    /// Returns the lower bound of the variables range.
	inline double              getLowerBound() {
		return(lowerBound);
	};

    /// Returns the upper bound of the variables range.
	inline double              getUpperBound() {
		return(upperBound);
	};
	
    /**
     * \brief Returns an LinguisticTerm object specified by its name.
     * 
     * @param name the name of the designated LinguisticTerm
     */
	const RCPtr<LinguisticTerm> findLT(std::string name);
    
    /**
     * \brief Returns the linguistic variable for the given name if existent or
     * throws FuzzyException otherwise.
     */
	static LinguisticVariable* getLV(std::string name);


private:
	//typedef map< long int, LinguisticTerm*, less< long int > > Termset;
	typedef std::list < LinguisticTerm* > Termset;
	Termset          terms;
	std::string      name;
	bool             destructorFlag;
	double           lowerBound; //c.f constructor
	double           upperBound;

	typedef std::map< std::string, LinguisticVariable*, std::less<std::string> > mapType;
	static mapType   lvMap;

};

#endif
