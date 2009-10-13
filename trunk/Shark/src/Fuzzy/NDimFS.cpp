/**
 * \file NDimFS.cpp
 *
 * \brief Base class for n-dimensional fuzzy sets
 * 
 * \authors Marc Nunkesser
 */


/* $log$ */


#include <Fuzzy/NDimFS.h>
#include <cassert>
#include <Fuzzy/Operators.h>
#include <Fuzzy/FuzzyException.h>



// empty destructor
NDimFS::~NDimFS() { };


// constructor(s)

NDimFS::NDimFS() {};


NDimFS::NDimFS(FuzzyArrayType & fat):
		components(fat) {};

// conversion

NDimFS::NDimFS(const RCPtr<FuzzySet> & fs) {
	components.clear();
	components.push_back(fs);
}

/*
double NDimFS::operator()(...) const
{
     //va_*** are C-Macros of the stdarg library dealing with variable parameter lists
    unsigned int i = getDimension();
    std::vector< double > vec;
    va_list ap;
    va_start( ap, i );
    for(unsigned int j=1; j<=i;j++)
       vec.push_back(va_arg(ap,double));
    va_end( ap );
    return((*this)(vec));
};
*/



//  NDimFS::operator FuzzySet&() const
// { if(getDimension()!=1)
//      throw(FuzzyException(21,"Conversion impossible since dimension of NDimFS is not one"));
//   return(*(components[0]));
// };


