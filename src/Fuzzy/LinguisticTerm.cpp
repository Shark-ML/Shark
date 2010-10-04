
/**
 * \file LinguisticTerm.cpp
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



#include "Fuzzy/LinguisticTerm.h"
#include "Fuzzy/LinguisticVariable.h"

//Instantiate static data member:
//long int LinguisticTerm::InstantiationCounter=0;

LinguisticTerm::LinguisticTerm(const std::string str): name(str)  {};

LinguisticTerm::LinguisticTerm(const std::string nameStr, const RCPtr<LinguisticVariable>& p): parent(p) {
	//Identifier=InstantiationCounter++;
	setName(nameStr);
	//add newly created object to LinguisticVar it belongs to
	if ((!parent)==0) parent->addTerm(this);
}

LinguisticTerm::~LinguisticTerm()

{
	if ((!parent)==0)
		try {
			parent->removeTerm(this);
		} catch (...) { /*ignore errors */ };
};


void LinguisticTerm::setName(const std::string inString) {
	name = inString;
}



void LinguisticTerm::setLinguisticVariable(const RCPtr<LinguisticVariable> &   inLV) {
	parent = inLV;
	parent->addTerm(this);
};
