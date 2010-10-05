
/**
 * \file SugenoIM.h
 *
 * \brief A Sugeno inference machine.
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


#ifndef SUGENOIM_H
#define SUGENOIM_H

#include <Fuzzy/InferenceMachine.h>
#include <Fuzzy/SingletonFS.h>

/**
 * \brief A Sugeno inference machine.
 */
class SugenoIM: public InferenceMachine {
public:


/**
 * \brief Constructor.
 * 
 * @param rb the associated rulebase
 */
	SugenoIM( RuleBase* rb = 0 );

/**
 * \brief Destructor
 */
	virtual ~SugenoIM();

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param inputType a vector of crisp values (an InputType)
 * @return the inference
 */
	double computeSugenoInference(const InputType inputType) const;

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param a the first crisp value
 * @param b the second crisp value
 * @return the inference
 */
	double computeSugenoInference(double a, double b) const;

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param a the first crisp value
 * @param b the second crisp value
 * @param c the third crisp value
 * @return the inference
 */
	double computeSugenoInference(double a, double b, double c) const;

/**
 * \brief Computes the Sugeno inference. 
 *
 * @param a the first crisp value
 * @param b the second crisp value
 * @param c the third crisp value
 * @param d the fourth crisp value
 * @return the inference
 */
	double computeSugenoInference(double a, double b, double c, double d) const;

private:

	virtual void             addToFile(double,std::ofstream &) const;
	virtual void             addToFile(double,double,std::ofstream &) const;
private:
	inline  virtual OutputType       buildTreeFast
	(RuleBase::BaseIterator & actual,
	 unsigned int remainingRules,
	 int conclusionNumber,
	 const InputType in) const {
		return( OutputType() );
	};
};

#endif
