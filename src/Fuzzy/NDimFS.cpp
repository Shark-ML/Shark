/**
 * \file NDimFS.cpp
 *
 * \brief Base class for n-dimensional fuzzy sets
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


