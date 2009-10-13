/**
 * \file SugenoRule.cpp
 *
 * \brief A Sugeno rule
 * 
 * \authors Marc Nunkesser
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
