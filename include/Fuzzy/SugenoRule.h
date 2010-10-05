
/**
 * \file SugenoRule.h
 *
 * \brief A Sugeno rule
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


#ifndef SUGENORULE_H
#define SUGENORULE_H


#include <Fuzzy/Rule.h>
#include <Fuzzy/FuzzyException.h>
#include <cassert>
#include <Fuzzy/RuleBase.h>


/**
 * \brief A Sugeno rule.
 * 
 * A Suegno rule its a rule with a linear combination of the imput values beeing 
 * the conclusion, like:<br>
 * <br>
 * <i>IF Tiredness IS high OR Soberness IS low THEN FitnessToDrive is 
 * 30-5*Tiredness+15*Soberness </i>
 */
class SugenoRule : public Rule {
public:
// We override the typedef in rule. A sugeno conclusion represents
// the coefficients for a linear combination of the
// inputs. Thus the length of the vector must be
// #inputs + 1


/**
 * \brief Default constructor.
 *  
 */
	inline SugenoRule(Connective c=AND, RuleBase * belongsTo = NULL): Rule(c,belongsTo) {}

	typedef std::vector<double>      ConclusionType;
	
	
/**
 * \brief Set the conclusion given a ConclusionType(the vector of coefficients for the liner combination).
 * 
 * @param cT the ConclusionType (a vector<double>) 
 */
	void setConclusion(ConclusionType & cT);
	
/**
 * \brief Set the conclusion given three coenfficients for the linear combination.
 * 
 * @param a  first coefficient for the linear combination
 * @param b  secound coefficient for the linear combination
 * @param c  third coefficient for the linear combination
 */	
	void setConclusion( double a, double b, double c);
	
/**
 * \brief Return the vector of coefficients for the linear combination of the conclusion.
 * 
 * @return the vector of coefficients for the liner combination of the conclusion
 */
	inline ConclusionType *     getConclusion() {
		return(&sugenoConclusion);
	};
	
/**
 * \brief Calculate the consequence (i.e. the resulting activation of the conclusion given the input).
 * 
 * @return the activation of the conclusion
 */	
	double                      calculateConsequence(const std::vector<double> & Inputs);

// override memberfunction with "wrong type".
// Do NOT use the following function!
	inline void                 addConclusion(LinguisticTerm & lt){
		throw(FuzzyException(12,"Sugeno rules do not have linguistic terms in their conlusion."));
	};
private:
	ConclusionType              sugenoConclusion;


};


#endif
