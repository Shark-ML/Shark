/*!
 * 
 *
 * \brief       Example for examining characteristics of the CMA with the help of
 * the Probe framework.
 * 
 * 
 *
 * \author      tvoss
 * \date        -
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;
using namespace std;

#include <boost/property_tree/json_parser.hpp>

int main( int argc, char ** argv ) {

	// Results go here.
	ofstream results( "results.txt" );
	// Plotting commands (gnuplot) go here.
	ofstream plot( "plot.txt" );
	plot << "set key outside bottom center" << endl;
	plot << "set size square" << endl;
	plot << "set zeroaxis" << endl;
	plot << "set border 0" << endl;
	plot << "set xrange [-4:4]" << endl;
	plot << "set yrange [-4:4]" << endl;
	// Adjust the floating-point format to scientific and increase output precision.
	results.setf( ios_base::scientific );
	results.precision( 10 );
	plot.setf( ios_base::scientific );
	plot.precision( 10 );

	// Instantiate both the problem and the optimizer.
//###begin<optimizer>
	Himmelblau hb;
	CMA cma;
	cma.init( hb );
//###end<optimizer>

	// Iterate the optimizer until a solution of sufficient quality is found.
	do{
		// Print error ellipses for covariance matrices.
		plot << "set object " 
		     << hb.evaluationCounter() + 1
		     << " ellipse center " 
		     << cma.mean()( 0 ) << ","
		     << cma.mean()( 1 ) << " size "
		     << cma.eigenValues()(0) * cma.sigma() * 2. << "," // times 2 because gunplot takes diameters as arguments 
		     << cma.eigenValues()(1) * cma.sigma() * 2. << " angle " 
		     << ::atan( cma.eigenVectors()( 1, 0 ) / cma.eigenVectors()( 0, 0 ) ) / M_PI * 180 << " front fillstyle empty border 2" << endl;
		
		// Report information on the optimizer state and the current solution to the console.
//###begin<results>
		results << hb.evaluationCounter() << " "	// Column 1
			<< cma.condition() << " "		// Column 2
			<< cma.sigma() << " "			// Column 3
			<< cma.solution().value << " ";		// Column 4
		copy(
		     cma.solution().point.begin(),				
		     cma.solution().point.end(),                // Column 5 & 6
		     ostream_iterator< double >( results, " " ) 
		     );
		copy( 
		     cma.mean().begin(),                        // Column 7 & 8
		     cma.mean().end(), 
		     ostream_iterator< double >( results, " " ) 
		      );
		results << endl;
//###end<results>

		// Do one CMA iteration/generation.
		cma.step( hb );

	} while( cma.solution().value> 1E-20 );

	// Write final result.
	// Print error ellipses for covariance matrices.
	plot << "set object " 
	     << hb.evaluationCounter() + 1
	     << " ellipse center " 
	     << cma.mean()( 0 ) << ","
	     << cma.mean()( 1 ) << " size "
	     << cma.eigenValues()(0) * cma.sigma() * 2. << "," // times 2 because gunplot takes diameters as arguments 
	     << cma.eigenValues()(1) * cma.sigma() * 2. << " angle " 
	     << ::atan( cma.eigenVectors()( 1, 0 ) / cma.eigenVectors()( 0, 0 ) ) / M_PI * 180  << " front fillstyle empty border 2" << endl;
	
	// Report information on the optimizer state and the current solution to the console.
	results << hb.evaluationCounter() << " "	// Column 1
		<< cma.condition() << " "		// Column 2
		<< cma.sigma() << " "			// Column 3
		<< cma.solution().value << " ";		// Column 4
	copy(
	     cma.solution().point.begin(),				
	     cma.solution().point.end(),                // Column 5 & 6
	     ostream_iterator< double >( results, " " ) 
	     );
	copy( 
	     cma.mean().begin(),                        // Column 7 & 8
	     cma.mean().end(), 
	     ostream_iterator< double >( results, " " ) 
	      );
	results << endl;

	//plot << "plot 'results.txt' every ::2 using 7:8 with lp title 'Population mean'" << endl;
	plot << "plot 'results.txt' using 7:8 with lp title 'Population mean'" << endl;

	return( EXIT_SUCCESS );	
}
