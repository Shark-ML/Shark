
/**
 * \file LinguisticVariable.cpp
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

#ifndef LINGUISTICVARIABLE_CPP
#define LINGUISTICVARIABLE_CPP

#include "Fuzzy/LinguisticVariable.h"

#include "Fuzzy/LinguisticTerm.h"

#include <assert.h>

// initialize the static data member
// This is a map of all Linguistic Variables which are declared.
// Its function is to be able to access all Linguistic Variables via
// static mf getLV

LinguisticVariable::mapType LinguisticVariable::lvMap;

LinguisticVariable* LinguisticVariable::getLV(std::string name) {
	if (LinguisticVariable* result = lvMap[name]) {
		return(result);
	} else {
		throw(FuzzyException(22,"Linguistic Variable does not exist."));
	};
};


LinguisticVariable::~LinguisticVariable() {
	destructorFlag = true; //cf removeTerm
	mapType::iterator itM;
	itM = lvMap.find(this->name);
	lvMap.erase(itM);
};


LinguisticVariable::LinguisticVariable(const std::string n,double l, double u): name(n),lowerBound(l),upperBound(u) {
	destructorFlag = false;
	lvMap[n]=this;
};


std::string LinguisticVariable::getName() const {
	return(name);
};

void LinguisticVariable::setName(const std::string instring) {
	name = instring;
};

void LinguisticVariable::setBounds(double lower, double upper) {
	lowerBound = lower;
	upperBound = upper;
};

void LinguisticVariable::addTerm(LinguisticTerm * lt) {
	// terms.push_back(const_cast<RCPtr<LinguisticTerm> &>(lt));
	terms.push_back(lt);
};

void LinguisticVariable::removeTerm(LinguisticTerm * lt) {
	if (!destructorFlag)
		terms.remove(lt); //for  lists
	// This is meant to avoid an infinite loop when the destructor calls the
	// destructors of all associated Linguistic Terms, which call
	// removeTerm() in this class, which normally destroys the
	// associated element, thus calls its destructor and so on.
};


const RCPtr<LinguisticTerm> LinguisticVariable::getTerm(int whichOne) {
	if ((whichOne<0) || (whichOne >= getNumberOfTerms())) {
		throw(FuzzyException(3,"Index of Linguistic Term out of Bounds"));
	} else {
		Termset::const_iterator it;
		it = terms.begin();
		for (int i=0;i<whichOne;i++) it++;
		//iterate over list till wichOne-th Element is reached.
		return((*it));
	}
	;//second
};

const RCPtr<LinguisticTerm> LinguisticVariable::findLT(std::string name) {
	Termset::const_iterator itT;
	//assert(terms.size()>0);
	for (itT=terms.begin(); itT!=terms.end() ;++itT) {
		if ((*itT)->getName()==name)
			break;
	};
	if ((*itT)->getName()==name) {
		return(*itT);
	} else {
		throw(FuzzyException(17,"There is no associated Linguistic Term with the given Name"));
	};
	std::cerr << "There is no associated Linguistic Term with the given Name" << std::endl;
}

#endif
