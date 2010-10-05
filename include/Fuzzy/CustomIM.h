/**
 * \file CustomIM.h
 *
 * \brief An user defined inference machine
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


#ifndef CUSTOMINFERENCEMACHINE_H
#define CUSTOMINFERENCEMACHINE_H


#include <Fuzzy/InferenceMachine.h>
#include <Fuzzy/Implication.h>


/**
 * \brief An user defined inference machine.
 *
 * This class enables the user to configurate an inference by his own.
 * An inference here is given as a sup-min composition of implication and
 * premise. 
 * \f[
 *      \mu(y) = \sup_{x} min(\mu_1(x), \mu_2(x,y))
 * \f]
 * Where \f$\mu_1\f$ is the premise and \f$\mu_2\f$ is the implication function.
 * Thus the inference mechanism is influenced by the implication
 * choosen by the user.
 */
class CustomIM: public InferenceMachine {
public:

/**
 * \brief Constructor
 * 
 * @param rb the associated rulebase
 * @param im the type of implication
 */
	CustomIM(RuleBase * rb ,Implication::ImplicationType im);

/**
 * \brief Destructor
 */
	virtual ~CustomIM();
	
private:
	Implication::ImplicationType  usedImplication;
	OutputType buildTreeFast (	RuleBase::BaseIterator & actual,
	                           unsigned int remainingRules,
	                           int conclusionNumber,
	                           const InputType in) const;

};


#endif
