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
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
  // Implementation of the CMA-ES
 #include <shark/Algorithms/DirectSearch/CMA.h>
 // Access to benchmark functions
 #include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark;
using namespace std;
		
int main( int argc, char ** argv ) {

	// Adjust the floating-point format to scientific and increase output precision.
	cout.setf( ios_base::scientific );
	cout.precision( 10 );

	// Instantiate the problem.
	benchmarks::Sphere sphere( 2 );
	// Initialize the optimizer for the objective function instance.
	CMA cma;
	cma.setInitialSigma(0.1);// Explicitely set initial global step size.
	sphere.init();
	cma.init( sphere, sphere.proposeStartingPoint()); 

	// Iterate the optimizer until a solution of sufficient quality is found.
	do {
		cma.step( sphere );

		// Report information on the optimizer state and the current solution to the console.
		cout << sphere.evaluationCounter() << " " << cma.solution().value << " " << cma.solution().point << " " << cma.sigma() << endl;
	} while(cma.solution().value > 1E-20 );	
}
