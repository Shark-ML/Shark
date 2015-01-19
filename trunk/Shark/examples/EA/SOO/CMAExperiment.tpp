/*!
 * 
 *
 * \brief       Example of an experiment using the CMA-ES on several benchmark functions
 *
 * \author      O.Krause
 * \date        2014
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
//###begin<includes>
// Implementation of the CMA-ES
#include <shark/Algorithms/DirectSearch/CMA.h>
// Access to benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/Rosenbrock.h>
#include <shark/ObjectiveFunctions/Benchmarks/Cigar.h>
#include <shark/ObjectiveFunctions/Benchmarks/Discus.h>
#include <shark/ObjectiveFunctions/Benchmarks/Ellipsoid.h>
		
using namespace shark;
//###end<includes>


int main( int argc, char ** argv ) {

//###begin<parameters>
	std::size_t numDimensions = 10; //dimensions of the objective functions
	std::size_t numTrials = 100; // how often the optimization is repeated
	std::size_t recordingInterval = 20; //we want to record after some multiple of this
	std::size_t numIterations = 20*recordingInterval; //number of iterations to perform
//###end<parameters>
	
//###begin<functions>
	//assortment of test functions
	typedef boost::shared_ptr<SingleObjectiveFunction > Function;
	std::vector<Function > functions;
	functions.push_back(Function(new Rosenbrock(numDimensions)));
	functions.push_back(Function(new Cigar(numDimensions)));
	functions.push_back(Function(new Discus(numDimensions)));
	functions.push_back(Function(new Ellipsoid(numDimensions)));
//###end<functions>	
	
//###begin<optimization>
	RealMatrix meanPerformance(functions.size(), numIterations/recordingInterval+1,0.0);
	for(std::size_t f = 0; f != functions.size(); ++f){
		for(std::size_t trial = 0; trial != numTrials; ++trial){
			//print progress
			std::cout<<"\r" <<functions[f]->name() <<": "<<trial<<"/"<<numTrials<<std::flush;
			//create and initialize the optimizer
			CMA cma;
			cma.init( *functions[f] );
			
			//record value
			meanPerformance(f,0) += cma.solution().value; 
			
			//optimize
			for(std::size_t i = 1; i <= numIterations; ++i){
				cma.step(*functions[f]);
				if(i % recordingInterval == 0){
					meanPerformance(f,i / recordingInterval) += cma.solution().value; 
				}
			}
		}
	}
	meanPerformance /= numTrials;
//###end<optimization>

//###begin<print>
	std::cout<<"\r# Iteration ";
	for(std::size_t f = 0; f != functions.size(); ++f)
		std::cout<<functions[f]->name()<<" ";
	std::cout<<"\n";
	
	std::cout.precision( 7 );
	for(std::size_t i = 0; i != meanPerformance.size2();++i){
		std::cout<< i*recordingInterval<<" ";
		for(std::size_t f = 0; f != functions.size(); ++f){
			std::cout<<meanPerformance(f,i)<<" ";
		}
		std::cout<<"\n";
	}
//###end<print>
}
