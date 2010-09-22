//===========================================================================
/*!
 *  \file fonMO-CMA.cpp
 *
 *  \brief Simple benchmark of different hypervolume algorithms. 
 *
 *	Relies on the data files provided by the Walking Fish Group.
 *
 *  \author  Thomas Vo§
 *  \date    2008
 *
 *  \par Copyright (c) 2005:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      EALib
 *
*
*  <BR>
*
*
*  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#include <SharkDefs.h>
#include <MOO-EALib/Hypervolume.h>

#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>


// not thread safe but works under windows:
unsigned GlobalComparatorDimension;
int Comparator( const void * a, const void * b) {
	
	double * x = (double*) a;
	double * y = (double*) b;

	if( x[GlobalComparatorDimension-1] == y[GlobalComparatorDimension-1] )
		return( 0 );
	else if( x[GlobalComparatorDimension-1] < y[GlobalComparatorDimension-1] )
		return( -1 );

	return ( 1 );
}

int main( int argc, char ** argv ) {
	
	unsigned int noPoints 	= 40000;
	unsigned int dimension 	= 3;
	
	
	std::string dataFile( "ran.40000pts.3d.1" );
	if( argc > 1 )
		dataFile = argv[1];
		
	
	double * regUp = new double[dimension];
	std::fill( regUp, regUp + dimension, -MAXDOUBLE );
	
	double * points = new double[noPoints*dimension];
	
	double * p = points;
	
	std::ifstream in( dataFile.c_str() );
	if(!in) {
		std::cerr << "cannot open " << dataFile.c_str() << std::endl;
		exit(EXIT_FAILURE);
	}
	
	std::string line; double d; unsigned counter = 0;
	while( std::getline( in, line ) ) {
		if( counter >= noPoints )
			break;
		
		if( line.empty() )
			break;
		
		if( line == "#" )
			continue;
		
		std::stringstream ss( line );
		for( unsigned i = 0; i < dimension; i++ ) {
			ss >> d;
			p[i] = d;
			
			regUp[i] = Shark::max( regUp[i], d );
		}
		
		p += dimension;
		counter++;
	}
	
	for( unsigned i = 0; i < dimension; i++ )
		regUp[i] += 1.0;
	
	double volume = 0;

	GlobalComparatorDimension = dimension;
	qsort( points, noPoints, dimension * sizeof( double ), Comparator );
	
	clock_t clockStart, clockEnd;
	clockStart = clock();
	volume = overmars_yap( points, regUp, dimension, noPoints );
	clockEnd = clock();														
	std::cout << "Overmars / Yap Shark: " << volume << std::endl;
	std::cout << "\t " << ((double)(clockEnd - clockStart))/CLOCKS_PER_SEC << std::endl;
			
	
	return( EXIT_SUCCESS );
}
