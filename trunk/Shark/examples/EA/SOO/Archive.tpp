/*!
 * 
 *
 * \brief       Demonstration of the archive fitness function wrapper.
 * 
 * 
 *
 * \author      Tobias Glasmachers
 * \date        2013
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

#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>
#include <shark/ObjectiveFunctions/EvaluationArchive.h>


using namespace shark;
using namespace std;


int main() {
	cout.setf( ios_base::scientific );
	cout.precision( 10 );

	// Instantiate the problem
	Sphere sphere( 2 );
	sphere.setNumberOfVariables( 2 );

	// Create an archive object as a wrapper around the problem
	typedef EvaluationArchive< VectorSpace<double>, double> ArchiveType;
	typedef EvaluationArchive< VectorSpace<double>, double>::PointResultPairConstIterator ArchiveIteratorType;
	ArchiveType wrapper(&sphere);

	// Initialize the optimizer for the objective function instance.
	CMA cma;
	cma.init( wrapper );

	// Iterate the optimizer until a solution of sufficient quality is found.
	const double target = 1e-4;
	cout << "Optimize the sphere benchmark problem to target accuracy " << target << endl;
	do {
		// Note the use of the wrapper instead of the fitness function:
		cma.step( wrapper );

		// Report information on the optimizer state and the current solution to the console.
		cout << sphere.evaluationCounter() << " "
			<< cma.solution().value << " "
			<< cma.solution().point << " "
			<< cma.sigma() << endl;
	} while (cma.solution().value > target);

	// output archive contents (all visited search points)
	size_t N = wrapper.size();
	cout << endl;
	cout << "The archive contains " << N << " evaluated search points:" << endl;
	for (ArchiveIteratorType it=wrapper.begin(); it != wrapper.end(); ++it)
	{
		RealVector const& x = it->point;
		double fx = it->result;
		cout << "   f( " << x << " ) = " << fx << endl;
	}
}
