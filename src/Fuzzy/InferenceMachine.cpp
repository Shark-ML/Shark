
/* $log$ */
/**
 * \file InferenceMachine.cpp
 *
 * \brief An inference machine
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

#include <Fuzzy/InferenceMachine.h>

#include <Fuzzy/Rule.h>
#include <Fuzzy/ComposedFS.h>
#include <Fuzzy/ConstantFS.h>
#include <cassert>
#include <Fuzzy/Operators.h>
#include <Fuzzy/LinguisticVariable.h>
#include <Fuzzy/TrapezoidFS.h>

InferenceMachine::InferenceMachine(RuleBase * rbp): ruleBasePtr(rbp) {};

InferenceMachine::OutputType InferenceMachine::computeInference(const InputType in)  const {
	RCPtr<TrapezoidFS> trap;
	RuleBase::BaseIterator first = ruleBasePtr->getFirstIterator();
	OutputType Result=buildTreeFast(first,ruleBasePtr->getNumberOfRules()-1,((*first)->getConclusion()).size(),in);
	unsigned i =0;
	for (RuleBase::ConclIt f= ruleBasePtr->getFirstConclIt();f!=ruleBasePtr->getLastConclIt();++f,++i) {
		if (!(Result[i])) continue;
		trap = new TrapezoidFS((*f)->getLowerBound(),
		                       (*f)->getLowerBound(),
		                       (*f)->getUpperBound(),
		                       (*f)->getUpperBound());
		Result[i]= new ComposedFS(ComposedFS::MIN,
		                          Result[i],
		                          trap);
	};
	return(Result);
}

InferenceMachine::~InferenceMachine()
{
}

InferenceMachine::OutputType InferenceMachine::computeInference(double a)  const {
	std::vector<double> v( 1 );
	v[0] = a;
	return( computeInference( v ) );
}

InferenceMachine::OutputType InferenceMachine::computeInference(double a, double b)  const {
	std::vector<double> v( 2 );
	v[0] = a;
	v[1] = b;
	return( computeInference( v ) );
}

InferenceMachine::OutputType     InferenceMachine::computeInference(double a, double b, double c)  const {
	std::vector<double> v( 3 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	return( computeInference( v ) );
}

InferenceMachine::OutputType     InferenceMachine::computeInference(double a, double b, double c, double d)  const {
	std::vector<double> v( 4 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	v[3] = d;
	return( computeInference( v ) );
}

/*
InferenceMachine::OutputType   InferenceMachine::computeInference(...) const throw(FuzzyException)
{   unsigned int i = ruleBasePtr->getNumberOfInputs();
    std::vector< double > vec;
    va_list ap;
    va_start(ap, i); // ignore warning
    for(unsigned int j=1; j<=i;j++)
       vec.push_back(va_arg(ap, double));
    va_end( ap );
    return(computeInference(vec));
};
*/

void InferenceMachine::characteristicCurve( const std::string fileName, const long int resolution ) const {
	double lowerI,upperI,lowerJ,upperJ;
	double stepI,stepJ;
	double i,j;
	RuleBase::FormatIterator fi;

	std::ofstream dataFile(fileName.c_str(),std::ios::out);
	if (!dataFile) {
		throw(FuzzyException(10,"Cannot write to disk"));
	};
	switch (ruleBasePtr->getNumberOfInputs()) {
	case 1:
		fi = ruleBasePtr->getFirstFormatIterator();
		lowerI=(*fi)->getLowerBound();
		upperI=(*fi)->getUpperBound();
		assert(lowerI<upperI);
		stepI = (upperI-lowerI)/resolution;
		for (i = lowerI;i<= upperI;i+=stepI) {
			addToFile(i,dataFile);
		};
		break;
	case 2:
		fi = ruleBasePtr->getFirstFormatIterator();
		lowerI=(*fi)->getLowerBound();
		upperI=(*fi)->getUpperBound();
		assert(lowerI<upperI);
		stepI = (upperI-lowerI)/resolution;
		fi++;
		lowerJ=(*fi)->getLowerBound();
		upperJ=(*fi)->getUpperBound();
		stepJ = (upperJ-lowerJ)/resolution;
		assert((stepJ>0)&&(stepI>0));
		for (i = lowerI;i<= upperI;i+=stepI) {
			for (j= lowerJ; j<= upperJ;j+=stepJ) {
				addToFile(i,j,dataFile);
			};
			dataFile << std::endl;
		};
		break;
	default:
		throw(FuzzyException(14,"Dimension of rules make visualization impossible"));
		break;
	};
	dataFile.close();
};



void InferenceMachine::addToFile(double i,std::ofstream & dataFile) const {
	std::vector<double> v( 1 );
	v[0] = i;
	dataFile << i << " " << (computeInference(v)[0])->defuzzify() << std::endl;
}

void InferenceMachine::addToFile(double i,double j,std::ofstream & dataFile) const {
	std::vector<double> v( 2 );
	v[0] = i;
	v[1] = j;
	dataFile << (computeInference(v)[0])->defuzzify() << " ";
}









