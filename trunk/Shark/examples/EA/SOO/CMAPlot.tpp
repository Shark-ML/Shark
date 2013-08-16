/**
 *
 * \brief Example for examining characteristics of the CMA with the help of
 * the Probe framework.
 * 
 * \author tvoss
 *  
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 *  
 */

#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;
using namespace std;

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/units/systems/si.hpp>

#include <limits>

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
		     << 360 + 360 * ::atan( cma.eigenVectors()( 0, 1 ) / cma.eigenVectors()( 0, 0 ) ) * 1./(2*M_PI) << " front fillstyle empty border 2" << endl;
		//<< 90 * ::atan( eigenVectors( 0, 1 ) / eigenVectors( 0, 0 ) ) * 2./M_PI << " front fillstyle empty border 2" << endl; // CI: Thomas original version, which may be correct, but I did not understand it directly
		
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
	     << 360 + 360 * ::atan( cma.eigenVectors()( 0, 1 ) / cma.eigenVectors()( 0, 0 ) ) * 1./(2*M_PI) << " front fillstyle empty border 2" << endl;
	
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
