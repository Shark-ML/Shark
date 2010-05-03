
/**
 * \file LinguisticTerm.cpp
 *
 * \brief A single linguistic term
 * 
 * \author Marc Nunkesser
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
