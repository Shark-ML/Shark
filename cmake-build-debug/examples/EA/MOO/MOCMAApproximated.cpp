/*!
 * 
 *
 * \brief       Example for running the approximated hypervolume MO-CMA-ES on an exemplary benchmark function.

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
// Implementation of the MO-CMA-ES
#include <shark/Algorithms/DirectSearch/MOCMA.h>
// Access to benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
		
int main( int argc, char ** argv ) {

	// Adjust the floating-point format to scientific and increase output precision.
	std::cout.setf( std::ios_base::scientific );
	std::cout.precision( 10 );

	// Instantiate both the problem and the optimizer.
	shark::benchmarks::DTLZ2 dtlz2;
	dtlz2.setNumberOfObjectives( 5 );
	dtlz2.setNumberOfVariables( 25 );
	

	shark::MOCMA mocma;
	mocma.mu() = 120;
	mocma.indicator().useApproximation(true);
	mocma.indicator().approximationDelta() = 0.05;
	mocma.indicator().setReference(shark::RealVector(dtlz2.numberOfObjectives(),11));
	// Initialize the optimizer for the objective function instance.
	dtlz2.init();
	mocma.init( dtlz2 );

	// Iterate the optimizer
	while( dtlz2.evaluationCounter() < 2000 ) {
		mocma.step( dtlz2 );
		std::cout<<dtlz2.evaluationCounter()<<std::endl;
	}

	// Print the optimal pareto front
	for( std::size_t i = 0; i < mocma.solution().size(); i++ ) {
		for( std::size_t j = 0; j < dtlz2.numberOfObjectives(); j++ ) {
			std::cout<< mocma.solution()[ i ].value[j]<<" ";
		}
		std::cout << std::endl;
	}
}
