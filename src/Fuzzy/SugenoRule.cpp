/**
 * \file SugenoRule.cpp
 *
 * \brief A Sugeno rule
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


#include <Fuzzy/SugenoRule.h>



// pre: Conclusion has already been set and input has correct size
double SugenoRule::calculateConsequence(const std::vector<double> & inputs) {
	assert(inputs.size()==((ruleBasePtr->getNumberOfInputs())));
	// assert(sugenoConclusion.size()==inputs.size()+	1);
	double sum = sugenoConclusion[0];
	for (unsigned int i = 0; i<inputs.size()+1;) {
// What the hell ... ??
// 		sum+=inputs[i]*sugenoConclusion[++i];

		// let's assume this:
		double tmp = inputs[i];
		i++;
		sum += tmp * sugenoConclusion[i];
	};
	return(sum);
};

void SugenoRule::setConclusion( double x, double y, double z ) {
	sugenoConclusion.resize( 3 );
	sugenoConclusion[0] = x;
	sugenoConclusion[1] = y;
	sugenoConclusion[2] = z;
};


void SugenoRule::setConclusion(ConclusionType & s) {
	assert(s.size()==((ruleBasePtr->getNumberOfInputs())+1));
	sugenoConclusion = s;
};

/*
void SugenoRule::setConclusion(...)
{   unsigned int i = (ruleBasePtr->getNumberOfInputs())+1;
    sugenoConclusion.clear();
    va_list ap;
    va_start(ap, i); //ignore warning
    for(unsigned int j=1; j<=i;j++)
       sugenoConclusion.push_back(va_arg(ap, double));
    va_end( ap );
}
*/
