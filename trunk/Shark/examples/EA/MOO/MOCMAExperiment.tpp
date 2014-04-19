/*!
 * 
 *
 * \brief       Example of an expriment using the MO-CMA-ES on several benchmark functions
 *
 * \author      O.Krause
 * \date        2014
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
// Implementation of the MO-CMA-ES
#include <shark/Algorithms/DirectSearch/MOCMA.h>
// Access to benchmark functions
#include <shark/ObjectiveFunctions/Benchmarks/ZDT1.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT2.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT3.h>
#include <shark/ObjectiveFunctions/Benchmarks/ZDT6.h>
		
using namespace shark;
//###end<includes>

//###begin<hypervolume>
//functor returning the value vector of a solution object
struct PointExtractor{
	template<class T>
	RealVector const& operator()(T const& arg)const{
		return arg.value;
	}
};
template<class Solution>
double hypervolume( Solution const& solution){
	// the reference point (11,11).
	RealVector referencePoint(2,11);
	//instance of the hypervolume calculator
	HypervolumeCalculator hypervolume;
	return hypervolume(PointExtractor(),solution,referencePoint);
}
//###end<hypervolume>


int main( int argc, char ** argv ) {

//###begin<parameters>
	std::size_t frontSize = 10; //number of points that approximate the front
	std::size_t numDimensions = 10; //dimensions of the objective functions
	std::size_t numTrials = 10; // how often the optimization is repeated
	std::size_t recordingInterval = 20; //we want to record after some multiple of this
	std::size_t numIterations = 20*recordingInterval; //number of iterations to perform
//###end<parameters>
	
//###begin<functions>
	//assortment of test functions
	typedef boost::shared_ptr<MultiObjectiveFunction> Function;
	std::vector<Function > functions;
	functions.push_back(Function(new ZDT1(numDimensions)));
	functions.push_back(Function(new ZDT2(numDimensions)));
	functions.push_back(Function(new ZDT3(numDimensions)));
	functions.push_back(Function(new ZDT6(numDimensions)));
//###end<functions>	
	
//###begin<optimization>
	RealMatrix meanVolumes(functions.size(), numIterations/recordingInterval+1,0.0);
	for(std::size_t f = 0; f != functions.size(); ++f){
		for(std::size_t trial = 0; trial != numTrials; ++trial){
			//print progress
			std::cout<<"\r" <<functions[f]->name() <<": "<<trial<<"/"<<numTrials<<std::flush;
			//create and initialize the optimizer
			MOCMA mocma;
			mocma.mu() = frontSize;
			mocma.init( *functions[f] );
			
			//record and hypervolume of initial solution
			meanVolumes(f,0) += hypervolume(mocma.solution()); 
			
			//optimize
			for(std::size_t i = 1; i <= numIterations; ++i){
				mocma.step(*functions[f]);
				if(i % recordingInterval == 0){
					meanVolumes(f,i / recordingInterval) += hypervolume(mocma.solution()); 
				}
			}
		}
	}
	meanVolumes /= numTrials;
//###end<optimization>

//###begin<print>
	std::cout<<"\r# Iteration ";
	for(std::size_t f = 0; f != functions.size(); ++f)
		std::cout<<functions[f]->name()<<" ";
	std::cout<<"\n";
	
	std::cout.precision( 7 );
	for(std::size_t i = 0; i != meanVolumes.size2();++i){
		std::cout<< i*recordingInterval<<" ";
		for(std::size_t f = 0; f != functions.size(); ++f){
			std::cout<<meanVolumes(f,i)<<" ";
		}
		std::cout<<"\n";
	}
//###end<print>
}
