#include <shark/Data/SparseData.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Models/NearestNeighborClassifier.h>
#include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
#include <shark/Algorithms/NearestNeighbors/SimpleNearestNeighbors.h>
#include <shark/Models/Trees/KDTree.h>
#include <shark/Models/Kernels/LinearKernel.h>

#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

int main(int argc, char **argv) {
	LabeledData<RealVector,unsigned int> data;
	importSparseData(data, "cod-rna",0,8192);
	
	LabeledData<RealVector,unsigned int> mnist;
	importSparseData(mnist, "mnist",0,8192);
	//~ {
	//~ Timer time;
	//~ KDTree<RealVector> kdtree(data.inputs());
	//~ TreeNearestNeighbors<RealVector,unsigned int> algorithmKD(data,&kdtree);
	//~ NearestNeighborClassifier<RealVector> model(&algorithmKD, 10);
	//~ ZeroOneLoss<> loss;
	//~ double error = loss(data.labels(),model(data.inputs()));
	//~ double time_taken = time.stop();
		
	//~ cout <<  "kdtree: "<< time_taken <<" "<< error<<std::endl;
	//~ }
	
	{
	Timer time;
	LinearKernel<RealVector> euclideanKernel;
	SimpleNearestNeighbors<RealVector,unsigned int> simpleAlgorithm(data,&euclideanKernel);
	NearestNeighborClassifier<RealVector> model(&simpleAlgorithm, 10);
	ZeroOneLoss<> loss;
	double error = loss(data.labels(),model(data.inputs()));
	double time_taken = time.stop();
		
	cout <<  "brute-force: "<< time_taken <<" "<< error<<std::endl;
	}
	
	{
	Timer time;
	LinearKernel<RealVector> euclideanKernel;
	SimpleNearestNeighbors<RealVector,unsigned int> simpleAlgorithm(mnist,&euclideanKernel);
	NearestNeighborClassifier<RealVector> model(&simpleAlgorithm, 10);
	ZeroOneLoss<> loss;
	double error = loss(mnist.labels(),model(mnist.inputs()));
	double time_taken = time.stop();
		
	cout <<  "brute-force-mnist: "<< time_taken <<" "<< error<<std::endl;
	}
	
}