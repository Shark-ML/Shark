/*!
 * 
 *
 * \brief       Example for running CMA-ES on an exemplary benchmark function.

 * 
 *
 * \author      tvoss
 * \date        -
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
 //###begin<includes>
 // Implementation of the CMA-ES
 #include <shark/Algorithms/DirectSearch/CMA.h>
 // Access to benchmark functions
 #include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
//###end<includes>

using namespace shark;
using namespace std;
		
int main( int argc, char ** argv ) {

	// Adjust the floating-point format to scientific and increase output precision.
	cout.setf( ios_base::scientific );
	cout.precision( 10 );

	// Instantiate both the problem
	//###begin<problem>
	Sphere sphere( 2 );
	sphere.setNumberOfVariables( 2 );
	//###end<problem>
	// Initialize the optimizer for the objective function instance.
	//###begin<optimizer>
	CMA cma;
	cma.init( sphere );
	//###end<optimizer>

	// Iterate the optimizer until a solution of sufficient quality is found.
	//###begin<train>
	do {

		cma.step( sphere );

		// Report information on the optimizer state and the current solution to the console.
		cout << sphere.evaluationCounter() << " "	
			<< cma.solution().value << " "
			<< cma.solution().point << " "
			<< cma.sigma() << endl;
	}while(cma.solution().value > 1E-20 );	
	//###end<train>
}
