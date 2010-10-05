/**
 * \file TriangularFS.cpp
 *
 * \brief FuzzySet with triangular membership function
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


/* $log$ */

#include <Fuzzy/TriangularFS.h>
#include <algorithm> // for min and max

TriangularFS::TriangularFS(double p1, double p2, double p3):
		a(p1), b(p2), c(p3) {};

TriangularFS::~TriangularFS() {};

double  TriangularFS::mu(double x) const {
	if (a>b) {
		throw(FuzzyException(1,"Illegal parameters for triangular MF --> a > b"));
	};
	if (b>c) {
		throw(FuzzyException(2,"Illegal parameters for triangular MF --> b > c"));
	};
	if (a == b && b == c)
		return(x == a);
	if (a == b)
		return((c-x)/(c-b)*(b<=x)*(x<=c));
	if (b == c)
		return((x-a)/(b-a)*(a<=x)*(x<=b));
	return(std::max(std::min((x-a)/(b-a), (c-x)/(c-b)), 0.0));


};


void TriangularFS::setParams(double x, double y, double z) {
	a=x;
	b= y;
	c=z;
};


