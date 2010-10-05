/**
 * \file RuleBase.cpp
 *
 * \brief A composite of rules 
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


#include "Fuzzy/RuleBase.h"
#include "Fuzzy/FuzzyException.h"


RuleBase::~RuleBase() {
	//cout<<"Destructor RuleBase "<<name<<endl;
	// RuleSet::const_iterator it;
	destructorFlag = true; //cf removeRule
	//     if(rules.size()>0)
	//       for(it=rules.begin(); it!=rules.end() ;++it) {delete((*it));};//second
};

RuleBase::RuleBase() {
	destructorFlag = false;
};

void RuleBase::addRule(const RCPtr<Rule>& r) {
	rules.push_back(const_cast< RCPtr<Rule> & >(r));
};

void RuleBase::removeRule(const RCPtr<Rule>& r) {
	if (!destructorFlag)
		rules.remove(r);
	// This is meant to avoid an infinite loop when the destructor calls the
	// destructors of all associated Rules, which call
	// removeRule() in this class, which normally destroys the
	// associated element, thus calls its destructor and so on.
};



// This method provides a simple but slow access to the rules.
// For faster access use the getFirsIterator and getLastIterator methods.

const RCPtr<Rule> RuleBase::getRule(int whichOne) {
	if ((whichOne<0) || (whichOne >= getNumberOfRules())) {
		throw(FuzzyException(7,"Index of Rules  out of Bounds"));
	} else {
		RuleSet::const_iterator it;
		it = rules.begin();
		for (int i=0;i<whichOne;i++) it++;
		//iterate over list till wichOne-th Element is reached.
		return((*it));
	}
	;//second
};


void RuleBase::addToInputFormat(const RCPtr<LinguisticVariable> &  l) {
	inputFormat.push_back(const_cast< RCPtr<LinguisticVariable> & >(l));
};


void RuleBase::addToOutputFormat(const RCPtr<LinguisticVariable> &  l) {
	outputFormat.push_back(const_cast< RCPtr<LinguisticVariable> & >(l));
};

void RuleBase::removeFromInputFormat(const RCPtr<LinguisticVariable>&  l) {
	inputFormat.remove(l);
};

void RuleBase::setInputFormat(inputTemplate & iT) {
	inputFormat =  iT;
};

const std::string RuleBase::print() const {
	std::string out;
	RuleSet::const_iterator it;
	if (rules.size()>0)
		for (it=rules.begin(); it!=rules.end() ;++it) {
			//std::cout << (*it)->printRule() << std::endl;
			out+=(*it)->printRule();
			out+="\n";
		};

	return(out);
};

void RuleBase::addToInputFormat(const RCPtr<LinguisticVariable> &  eins,
                                const RCPtr<LinguisticVariable> &  zwei) {
	addToInputFormat(eins);
	addToInputFormat(zwei);
}

void RuleBase::addToInputFormat(const RCPtr<LinguisticVariable> &  eins,
                                const RCPtr<LinguisticVariable> &  zwei,
                                const RCPtr<LinguisticVariable> &  drei) {
	addToInputFormat(eins);
	addToInputFormat(zwei);
	addToInputFormat(drei);
};

void RuleBase::addToInputFormat(const RCPtr<LinguisticVariable> &  eins,
                                const RCPtr<LinguisticVariable> &  zwei,
                                const RCPtr<LinguisticVariable> &  drei,
                                const RCPtr<LinguisticVariable> &  vier) {
	addToInputFormat(eins);
	addToInputFormat(zwei);
	addToInputFormat(drei);
	addToInputFormat(vier);
};


void RuleBase::addToOutputFormat(const RCPtr<LinguisticVariable> &  eins,
                                 const RCPtr<LinguisticVariable> &  zwei) {
	addToOutputFormat(eins);
	addToOutputFormat(zwei);
}
