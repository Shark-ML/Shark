/**
 * \file SugenoIM.cpp
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


#include <Fuzzy/SugenoIM.h>
#include <Fuzzy/SugenoRule.h>


SugenoIM::SugenoIM(RuleBase * rb): InferenceMachine(rb)
{ }

SugenoIM::~SugenoIM()
{ }


double SugenoIM::computeSugenoInference(const InputType input) const
// We calculate the inference according to Bothe p154 as
// sum(beta_i*y_i)/sum(beta_i)
// beta_i= Activation of Rule i
// y_i result of sugeno-consequence of rule i

{
	SugenoRule* actualRule;
	double nominator = 0;
	double denominator = 0;
	double temp;
	for (int i = 0;i<ruleBasePtr->getNumberOfRules();i++) {
		actualRule = (SugenoRule*) ruleBasePtr->getRule(i).operator->();
		temp = actualRule->Activation(input);
		nominator += temp*actualRule->calculateConsequence(input);
		denominator += temp;
	}
	if (denominator!=0)
		return(nominator/denominator);
	else
		throw(FuzzyException(13,"No rule was activated by the given input"));
};

double SugenoIM::computeSugenoInference(double a, double b) const {
	std::vector<double> v( 2 );
	v[0] = a;
	v[1] = b;

	return( computeSugenoInference( v ) );
}
double SugenoIM::computeSugenoInference(double a, double b, double c) const {
	std::vector<double> v( 3 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	return( computeSugenoInference( v ) );
}
double SugenoIM::computeSugenoInference(double a, double b, double c, double d) const {
	std::vector<double> v( 4 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	v[3] = c;
	return( computeSugenoInference( v ) );
}

/*
double SugenoIM::computeSugenoInference(...) const throw(FuzzyException)
{   unsigned int i = ruleBasePtr->getNumberOfInputs();
    std::vector< double > vec;
    va_list ap;
    va_start(ap, i); //ignore warning
    for(unsigned int j=1; j<=i;j++)
       vec.push_back(va_arg(ap, double));
    va_end( ap );
    return(computeSugenoInference(vec));
};
*/
void SugenoIM::addToFile(double i,std::ofstream & dataFile) const {
	// dataFile<<i<<" "<<computeSugenoInference(i)<<std::endl;
}

void SugenoIM::addToFile(double i,double j,std::ofstream & dataFile) const {
	//dataFile<<computeSugenoInference(i,j)<<std::endl;
}
