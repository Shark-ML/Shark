
/**
 * \file HomogenousNDimFS.cpp
 *
 * \brief A homogenous n-dimensional fuzzy set
 * 
 * \authors Marc Nunkesser
 */

/* $log$ */


#include <Fuzzy/HomogenousNDimFS.h>
#include <cassert>
#include <Fuzzy/Operators.h>



// empty destructor
HomogenousNDimFS::~HomogenousNDimFS() { };


// constructor
HomogenousNDimFS::HomogenousNDimFS(FuzzyArrayType & fat , Connective c):
		NDimFS(fat) {
	setConnective(c);
};

HomogenousNDimFS::HomogenousNDimFS(const RCPtr<FuzzySet> & fs):NDimFS(fs) {
	setConnective(AND); // choose arbitrary connective
};

double HomogenousNDimFS::operator()(const std::vector<double> & Inputs) const
// It is important to note that we allow the components to be empty,
// what makes the computation more difficult
{
	double temp;
	assert(Inputs.size()==components.size()); //parameters OK?
	double out; //value that will be returned
	std::vector< double >::const_iterator inputIt;
	FuzzyArrayType::const_iterator componentsIt;
	componentsIt = components.begin();
	inputIt = Inputs.begin();
	// find first non empty component:
	while (!(*componentsIt) && componentsIt!=components.end()) {
		++componentsIt;
		++inputIt;
	}
	// if there is one, calculate mu at point given by input
	if (!(!(*componentsIt))) {
		out = (**componentsIt)(*inputIt);
	} else {
		throw(FuzzyException(25,"Homogenous Fuzzy Set has no components"));
	};
	if (componentsIt==components.end()) {
		return(out);
	};
	++inputIt;
	++componentsIt;
	//Iterate over components and input values
	for (;inputIt!=Inputs.end();++inputIt,++componentsIt) { // out = connectiveFunc(out,(**componentsIt)(*inputIt));};
		if (!(*componentsIt)) continue;
		temp = (**componentsIt)(*inputIt);
		out = connectiveFunc(out,temp);
	};
	return(out);
}

double HomogenousNDimFS::operator()(double a ) const {
	std::vector<double> v( 1 );
	v[0] = a;
	return( (*this)(v) );
}

double HomogenousNDimFS::operator()(double a, double b ) const {
	std::vector<double> v( 2 );
	v[0] = a;
	v[1] = b;
	return( (*this)(v) );
}

double HomogenousNDimFS::operator()(double a, double b, double c ) const {
	std::vector<double> v( 3 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	return( (*this)(v) );
}

double HomogenousNDimFS::operator()(double a,double b, double c, double d ) const {
	std::vector<double> v( 4 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	v[3] = d;
	return( (*this)(v) );
}

/*
double HomogenousNDimFS::operator()(...) const
{
     //va_*** are C-Macros of the stdarg library dealing with variable parameter lists
    unsigned int i = getDimension();
    std::vector< double > vec;
    va_list ap;
    va_start(ap, i);
    for(unsigned int j=1; j<=i;j++)
       vec.push_back(va_arg(ap,double));
    va_end( ap );
    return((*this)(vec));
};
*/

void HomogenousNDimFS::setConnective(Connective c) {
	compoConnective = c;
	switch (compoConnective) {
	case AND :
		connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::minimum);
		break;
	case OR :
		connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::maximum);
		break;
	case PROD:
		connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::prod);
		break;
	case PROBOR:
		connectiveFunc = reinterpret_cast < double (*) (double,double)>(Operators::probor);
		break;
	};
}
