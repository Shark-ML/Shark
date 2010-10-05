
/**
 * \file Rule.cpp
 *
 * \brief  A rule which is composed of premise and conclusion
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


#include <Fuzzy/Rule.h>

#include <cassert>
#include <ctype.h>
#include <string.h>

#include <Fuzzy/ComposedLT.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/LinguisticTerm.h>
#include <Fuzzy/Operators.h>
#include <Fuzzy/NDimFS.h>
#include <Fuzzy/HomogenousNDimFS.h>

#include <Fuzzy/RuleBase.h>


// forward declaration:
char* nextToken(char* tokenPtr);

Rule::~Rule() {}; //empty destructor

Rule::Rule(Connective c,
           RuleBase * belongsTo,
           double _weight):
		ruleBasePtr(belongsTo),
		weight(_weight) {
	belongsTo->addRule(this);
	setConnective(c);
};

Rule::Rule(std::string RuleText,
           RuleBase * belongsTo,
           double _weight) :
		ruleBasePtr(belongsTo),
		weight(_weight) {

	belongsTo->addRule(this);

	// std::cout  << "w " << _weight << std::endl;
	// std::cout  << RuleText << std::endl;


	enum Mode {AND,OR,PROD,PROBOR,NONE,ERROR};//order is important cf connective
	LinguisticVariable* usedLV;
	RCPtr<LinguisticTerm> usedLT;
	bool premiseOver = false;

	Mode parseMode = NONE;
	char* tokenPtr;

	int len = RuleText.length();
	char* text = new char[len+1];
	RuleText.copy(text,len,0);
	text[len]=0; //add null terminator (!)
	tokenPtr = strtok(text," ");
	if (strcmp(tokenPtr,"IF\0")!=0) {
		throw(FuzzyException(15,"Wrong Format for Fuzzy Rule. Rule should begin with an 'IF'."));
	};
	do { //parse premise
		tokenPtr=nextToken(tokenPtr);
		usedLV = LinguisticVariable::getLV(tokenPtr);
		tokenPtr=nextToken(tokenPtr);
		if (strcmp(tokenPtr,"IS\0")!=0) {
			throw(FuzzyException(16,"Wrong Format for Fuzzy Rule. 'is' expected"));
		};
		tokenPtr=nextToken(tokenPtr);
		usedLT=( usedLV->findLT((std::string)tokenPtr)); //const cast
		addPremise(usedLT);
		//std::cout << "added " << tokenPtr << std::endl;
		tokenPtr=nextToken(tokenPtr);
		if (strcmp(tokenPtr,"THEN\0")==0) {
			premiseOver=true;
			break;
		};
		if (strcmp(tokenPtr,"AND\0")==0) {
			parseMode = ((parseMode==AND)||(parseMode==NONE)?AND:ERROR);
		};
		if (strcmp(tokenPtr,"OR\0")==0) {
			parseMode = ((parseMode==OR)||(parseMode==NONE)?OR:ERROR);
		};
		if (strcmp(tokenPtr,"PROD\0")==0) {
			parseMode = ((parseMode==PROD)||(parseMode==NONE)?PROD:ERROR);
		};
		if (strcmp(tokenPtr,"PROBOR\0")==0) {
			parseMode = ((parseMode==PROBOR)||(parseMode==NONE)?PROBOR:ERROR);
		};
	} while ((!premiseOver)&&(parseMode!=ERROR));

	if (parseMode==ERROR)
		throw(FuzzyException(18,"Wrong Format for Fuzzy Rule. AND, OR,PROD, PROBOR, THEN expected"));
	// for one element premises
	parseMode = (parseMode==NONE?AND:parseMode);
	// set connective
	setConnective((Connective)parseMode); //Rule::Connective
	//using the fact that the first four elements of parseMode and Connective are the same
	tokenPtr=nextToken(tokenPtr);
	do { // parse conclusion
		usedLV = LinguisticVariable::getLV(tokenPtr);
		tokenPtr=nextToken(tokenPtr);
		if (strcmp(tokenPtr,"IS\0")!=0) {
			throw(FuzzyException(16,"Wrong Format for Fuzzy Rule. 'is' expected"));
		};
		tokenPtr=nextToken(tokenPtr);
		usedLT=(usedLV->findLT(tokenPtr));
		addConclusion(usedLT);
		// std::cout << "added2conclusion " << tokenPtr << std::endl;
		tokenPtr=nextToken(tokenPtr);
	} while (tokenPtr!=NULL);
	delete [] text;

	// std::cout << printRule() << std::endl;

	// std::cout <<"parsing done"<< std::endl;
};

char* Rule::nextToken(char* tokenPtr) {
	if (tokenPtr!=NULL) {
		tokenPtr = strtok(NULL," ");
		return(tokenPtr);
	} else {
		throw(FuzzyException(16,"Wrong Format for Fuzzy Rule. Rule is incomplete."));
	};
};

void Rule::setRule(IORuleType & p ,Connective rC, ConclusionType & c)
// Synatctical correctness should be verified in further extensions
{
	premise = p;
	conclusion = c;
	ruleConnective = rC;
};




double Rule::Activation(const std::vector<double> & Inputs) const {
	// cout<<"Input:"<<Inputs.size()<<endl;
	// cout<<"premise"<<premise.size()<<endl;
	assert(Inputs.size()==premise.size()); //parameters OK?
	double out; //value that will be returned
	std::vector< double >::const_iterator inputIt;
	IORuleType::const_iterator premiseIt;
	premiseIt = premise.begin();
	inputIt = Inputs.begin();

	while (!(*premiseIt)) { // move to first non null premise
		if (inputIt==Inputs.end())
			throw(FuzzyException(23,"Premise is empty"));
		++inputIt;
		++premiseIt;
	};

	out = (**premiseIt)(*inputIt); // initialize out
	if (Inputs.size()==1) {
		return(out);
	};
	++inputIt;
	++premiseIt;
	//Iterate over premise and input values
	for (;inputIt!=Inputs.end();++inputIt,++premiseIt) {
		if (!(*premiseIt)) // if empty
			{} //skip
		else {
			out = connectiveFunc(out,(**premiseIt)(*inputIt));
		};
	};
	return(out);
};

void Rule::addPremise(const RCPtr<LinguisticTerm>& lt) {
// The general idea is to iterate over the inputFormat list of the
// associated Rule Base and the premise in parallel in order to find
// the "right place" for the input.
// This means that we want the premise to be in a normalized form:
// The order of the LTs must correspond to the order given by
// inputFormat in the Rule Base.
// Since there can be more than one LT of the same type, which is
// incompatible with this normalized form, we merge these
// terms to one sole LT using a ComposedLT.
// Thus the precondition for addPremise is:
// - the premise is already in normalized form, containig null pointers
//   where there is no LT of the demanded type.
// Postcondition:
// - the premise is still in normalized form, containing now the
//   additional LT given by the parameter lt.


	if (ruleBasePtr==NULL) {
		throw(FuzzyException(9,"Rule is actually associated to no rule Base"));
	};
	bool completed = false;
	RuleBase::FormatIterator formatIt = ruleBasePtr->getFirstFormatIterator();
	IORuleType::iterator premiseIt;
	if (premise.size()==0) {
		initializePremise();
	};
	for (premiseIt = premise.begin();premiseIt!=premise.end();++formatIt,++premiseIt)
		//does the type of the lt to add match the type of the actual place in the input format?
	{
		if (lt->getLinguisticVariable()==(*formatIt)) {
			completed = true; // the FS will now be added to the premise
			if (!(*premiseIt)) { // is this the first lt of this type in the premise?
				*premiseIt=const_cast<RCPtr<LinguisticTerm>&>(lt);
			} else {
				// there is already a LT of the same type. Construct a ComposedLT out
				// of these the existing and the new one.
				switch (ruleConnective) {
				case AND:
					*premiseIt = RCPtr<ComposedLT>(new ComposedLT(lt->getName(),
					                               (lt->getLinguisticVariable()),
					                               ComposedFS::MIN,
					                               const_cast<RCPtr<LinguisticTerm>&>(lt),
					                               *premiseIt));  //*premiseIt
					break;
				case OR:
					*premiseIt = RCPtr<ComposedLT>(new ComposedLT(lt->getName(),
					                               (lt->getLinguisticVariable()),
					                               ComposedFS::MAX,
					                               const_cast<RCPtr<LinguisticTerm>&>(lt),
					                               *premiseIt));
					break;
				case PROD:
					*premiseIt = RCPtr<ComposedLT>(new ComposedLT(lt->getName(),
					                               (lt->getLinguisticVariable()),
					                               ComposedFS::PROD,
					                               const_cast<RCPtr<LinguisticTerm>&>(lt),
					                               *premiseIt));
					break;
				case PROBOR:
					*premiseIt = RCPtr<ComposedLT>(new ComposedLT(lt->getName(),
					                               (lt->getLinguisticVariable()),
					                               ComposedFS::PROD,
					                               const_cast<RCPtr<LinguisticTerm>&>(lt),
					                               *premiseIt));
					break;
				default:
					throw(FuzzyException(8,"Unknown Connective in rule."));
				};
			}
		};
	};
	if (!completed) {
		throw(FuzzyException(11,"The type of the Linguistic Term does not match any type in the input format of the rule base"));
	}
};


void  Rule::addConclusion(const RCPtr<LinguisticTerm>& lt) {
	conclusion.push_back(const_cast<RCPtr<LinguisticTerm>&>(lt));
};

std::string Rule::printRule() const {
	std::string out = "IF ";
	RCPtr<LinguisticVariable> tempLV;
	const LinguisticTerm * tempLT;
	// premise
	for (unsigned int i = 0;i<premise.size();i++) {
		const LinguisticTerm * tempLT = dynamic_cast< const LinguisticTerm * >(premise[i].operator->()); // (!) this is a dirty trick -> returns a pointer to a FS
		if (tempLT==0)
			continue;
		tempLV = (tempLT->getLinguisticVariable());
		out += tempLV->getName()+ " IS " + tempLT->getName();
		std::cout << out << std::endl;
		if (i<premise.size()-1) {
			switch (ruleConnective) {
			case AND:
				out+=" AND ";
				break;
			case OR:
				out +=" OR ";
				break;
			case PROD:
				out +=" PROD ";
				break;
			case PROBOR:
				out +=" PROBOR ";
			default:
				throw(FuzzyException(8,"Unknown Connective in rule."));
			}
		};
	};
	// conclusion
	out +=" THEN ";
	std::cout << out << std::endl;
	std::cout << conclusion.size() << std::endl;
	tempLT = NULL;
	for (unsigned int i=0;i<conclusion.size();i++) {
		//tempLT = dynamic_cast< const LinguisticTerm * >(premise[i].operator->());
		tempLT = dynamic_cast< const LinguisticTerm * >(conclusion[i].operator->());
		if (tempLT==NULL) std::cout << "Hallo" << std::endl;
		assert(tempLT!=0); // cast OK?

		std::cout << tempLT->getName() << std::endl;

		tempLV = (tempLT->getLinguisticVariable());
		if (tempLT==0) std::cout << "gLV done" << std::endl;
		out += tempLV->getName()+ " IS " + tempLT->getName();
		std::cout << out << std::endl;
		if (i<premise.size()-1) out+= " ; ";
		std::cout << out << std::endl;
	};
	return(out);

	return("");

};

/*
double Rule::Activation(...) const
 {  //va_*** are C-Macros of the stdarg library dealing with variable parameter lists
    unsigned int i = ruleBasePtr->getNumberOfInputs();
    std::vector< double > vec;
    va_list ap;
    va_start(ap,i);
    for(unsigned int j=1; j<=i;j++)
       vec.push_back(va_arg(ap, double));
    va_end( ap );
    return(Activation(vec));
  }
*/

void Rule::initializePremise() {
	premise.clear();
	int upperBound = ruleBasePtr->getNumberOfInputs();
	for (int i = 0;i<upperBound;i++) {
		premise.push_back(NULL);
	};
};


void Rule::setConnective(Connective c) {
	ruleConnective = c;
	{
		switch (ruleConnective) {
		case AND : //  the embodyment of pure evil!
			connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::minimum);
			break;
		case OR :
			connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::maximum);
			break;
		case PROD:
			connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::prod);
			break;
		case PROBOR:
			connectiveFunc = reinterpret_cast<double (*) (double,double)>(Operators::probor);
			break;
		}
	};
};

NDimFS* Rule::getPremise() { // think about memory deallocation!
	HomogenousNDimFS* ndfs = new HomogenousNDimFS(premise,ruleConnective);
	return(ndfs);
};

double Rule::Activation(double in) const {
	std::vector< double > vec;
	vec.push_back(in);
	return(Activation(vec));
}

void Rule::addPremise(const RCPtr<LinguisticTerm> & a,
                      const RCPtr<LinguisticTerm> & b) {
	addPremise(a);
	addPremise(b);
}

void Rule::addPremise(const RCPtr<LinguisticTerm> & a,
                      const RCPtr<LinguisticTerm> & b,
                      const RCPtr<LinguisticTerm> & c) {
	addPremise(a);
	addPremise(b);
	addPremise(c);
}

void Rule::addPremise(const RCPtr<LinguisticTerm> & a,
                      const RCPtr<LinguisticTerm> & b,
                      const RCPtr<LinguisticTerm> & c,
                      const RCPtr<LinguisticTerm> & d) {
	addPremise(a);
	addPremise(b);
	addPremise(c);
	addPremise(d);
}
