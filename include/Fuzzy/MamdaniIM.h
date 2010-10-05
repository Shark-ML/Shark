
/**
 * \file MamdaniIM.h
 *
 * \brief A Mamdami inference machine
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


#ifndef MAMDANIINFERENCEMACHINE_H
#define MAMDANIINFERENCEMACHINE_H


// include files:


#include <Fuzzy/InferenceMachine.h>

/**
 * \brief A Mamdami inference machine.
 *
 * A Mamdami inference is given by 
 * \f$I(x,y)= min(x,y)\f$
 */
class MamdaniIM: public InferenceMachine {
public:
/**
 * \brief Constructor.
 * 
 * @param rb the associated rule base
 */
	MamdaniIM( RuleBase * rb );

/**
 * \brief Destructor
 */
	virtual ~MamdaniIM();

private:
	OutputType             buildTreeFast
	(RuleBase::BaseIterator & actual,
	 unsigned int remainingRules,
	 int conclusionNumber,
	 const InputType in) const;



//Build the tree of FuzzySets, whose evaluation yields the result of the c-th conclusion, starting with the a-th rule, and a ruleBase of length b.


};


#endif
