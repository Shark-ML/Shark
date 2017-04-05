#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculatorMD.h>
#include <shark/Algorithms/DirectSearch/Operators/Hypervolume/HypervolumeCalculatorMDWFG.h>

#include <shark/Core/Timer.h>
#include <shark/Core/Random.h>
#include <iostream>
using namespace shark;

std::vector<RealVector> createRandomFront(std::size_t numPoints, std::size_t numObj, double p){
	std::vector<RealVector> points(numPoints);
	for (std::size_t i = 0; i != numPoints; ++i) {
		points[i].resize(numObj);
		double norm = 0;
		double sum = 0;
		for(std::size_t j = 0; j != numObj; ++j){
			points[i](j) = 1- random::uni(0.0, 1.0-sum);
			sum += 1-points[i](j);
			norm += std::pow(points[i](j),p);
		}
		norm = std::pow(norm,1/p);
		points[i] /= norm;
	}
	return points;
}

int main(int argc, char **argv) {
	
	
	random::seed(42);
	for(std::size_t dim = 4; dim != 9; ++dim){
		std::cout<<"dimensions = " <<dim<<std::endl;
		RealVector reference(dim,1.0);
		for(unsigned int numPoints = 10; numPoints != 110; numPoints +=10){
			auto set = createRandomFront(numPoints,dim,2);
			
			HypervolumeCalculatorMD algorithm1;
			HypervolumeCalculatorMDWFG algorithm2;
			
			double val1= 0;
			double stop1 = 0;
			{
				Timer time;
				val1 = algorithm1(set, reference);
				stop1 = time.stop();
			}
			double val2= 0;
			double stop2 = 0;
			{
				Timer time;
				val2 = algorithm2(set, reference);
				stop2 = time.stop();
			}
			std::cout<<numPoints<<"\t"<<stop1<<"\t"<<stop2<<"\t"<<val1-val2<<"\t"<<std::endl;
		}
		std::cout<<std::endl;
	}
}