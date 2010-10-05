/**
 * \file ComposedNDimFS.cpp
 *
 * \brief A composed n-dimensional FuzzySet
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


#include <Fuzzy/ComposedNDimFS.h>

#include <assert.h>

ComposedNDimFS::ComposedNDimFS(
    const RCPtr<NDimFS>&  left,
    const RCPtr<NDimFS>&  right,
    double (*uF)(double,double) ):
		leftOperand(left),rightOperand(right),userDefinedOperator(reinterpret_cast< double (*) (double,double)>(uF)) {
	if (left->getDimension() != right->getDimension())
		throw(FuzzyException(24,"The two n-dimensional Fuzzy sets a ComposedNDimFS consists of must have the same dimension"));
}



double ComposedNDimFS::operator()(const std::vector<double>&  x) const {
	double arg1 = (*leftOperand)(x);
	double arg2 = (*rightOperand)(x);
	return((*userDefinedOperator)(arg1,arg2));
}

double ComposedNDimFS::operator()(double a ) const {
	std::vector<double> v( 1 );
	v[0] = a;
	return( (*this)(v) );
}

double ComposedNDimFS::operator()(double a, double b ) const {
	std::vector<double> v( 2 );
	v[0] = a;
	v[1] = b;
	return( (*this)(v) );
}

double ComposedNDimFS::operator()(double a, double b, double c ) const {
	std::vector<double> v( 3 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	return( (*this)(v) );
}

double ComposedNDimFS::operator()(double a,double b, double c, double d ) const {
	std::vector<double> v( 4 );
	v[0] = a;
	v[1] = b;
	v[2] = c;
	v[3] = d;
	return( (*this)(v) );
}


// double ComposedNDimFS::operator()(double d) const {
//   vector< double > vec;
//   vec.push_back(d);
//   return((*this)(vec));
// }

/*
double ComposedNDimFS::operator()(...) const {
     //va_*** are C-Macros of the stdarg library dealing with variable parameter lists
    unsigned int i = getDimension();
    std::vector< double > vec;
    va_list ap;
    va_start(ap, i);
    for(unsigned int j=1; j<=i;j++)
       vec.push_back(va_arg(ap,double));
    va_end( ap );
    return((*this)(vec));
}
*/

ComposedNDimFS::operator RCPtr<ComposedFS>() {
	assert((leftOperand->getDimension() == 1) && (rightOperand->getDimension() ==1));
	RCPtr<ComposedFS> out(new ComposedFS(ComposedFS::USER,(*leftOperand)[0],(*rightOperand)[0],userDefinedOperator));
	return(out);
};
