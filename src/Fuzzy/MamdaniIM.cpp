/**
 * \file MamdaniIM.cpp
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


#include <Fuzzy/MamdaniIM.h>
#include <Fuzzy/Rule.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/ConstantFS.h>
#include <Fuzzy/Operators.h>


MamdaniIM::MamdaniIM(RuleBase * rbp): InferenceMachine(rbp)
{ }

MamdaniIM::~MamdaniIM()
{ }


MamdaniIM::OutputType MamdaniIM::buildTreeFast
(RuleBase::BaseIterator & actual,
 unsigned int remainingRules,
 int conclusionNumber,
 const  InputType in) const {

	RCPtr<Rule> actualRule = *actual;
	const Rule::ConclusionType& actualConclusion = actualRule->getConclusion();
	OutputType conclusionFragment(conclusionNumber);
	OutputType minNode(conclusionNumber);
	OutputType maxNode(conclusionNumber);
	RCPtr<ConstantFS> betaNode(new ConstantFS(actualRule->Activation(in)));

	for (int i = 0;i<conclusionNumber;i++) {
		conclusionFragment[i] = (actualConclusion)[i];
		minNode[i] = Operators::min(betaNode,conclusionFragment[i]);
	};

	if (remainingRules==0) {
		for (int i = 0;i<conclusionNumber;i++) minNode[i]->scale(actualRule->getWeight());
		return (minNode);
	} else {
		for (int i = 0;i<conclusionNumber;i++)  {
			minNode[i]->scale(actualRule->getWeight());
			maxNode[i]= new ComposedFS(ComposedFS::MAX,minNode[i],(buildTreeFast(++actual,--remainingRules,conclusionNumber,in))[i]);
		};
		return(maxNode);
	};
};
