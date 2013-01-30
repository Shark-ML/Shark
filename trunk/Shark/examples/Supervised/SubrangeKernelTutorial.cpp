#include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/WeightedSumKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/SubrangeKernel.h>
#include <shark/Rng/GlobalRng.h>

using namespace shark;

int main()
{
	std::cout << "\n ----- Starting MklKernel normalization demo ---- \n\n" << std::flush;
	
	std::size_t num_dims = 9;
	std::size_t num_points = 200;
	std::vector<RealVector> input(num_points);
	RealVector v(num_dims);
	for ( std::size_t i=0; i<num_points; i++ ) {
		for ( std::size_t j=0; j<num_dims; j++ ) {
			v(j) = Rng::uni(-1,1);
		}
		input[i] = v;
	}
	UnlabeledData<RealVector> data(input);
	
	DenseRbfKernel   	  basekernel1(0.1);
	DenseLinearKernel      basekernel2;
	DensePolynomialKernel  basekernel3(2, 1.0);
	
	std::vector< AbstractKernelFunction<RealVector> * > kernels;
	kernels.push_back(&basekernel1);
	kernels.push_back(&basekernel2);
	kernels.push_back(&basekernel3);
	
	std::vector< std::pair< std::size_t, std::size_t > > frs;
	frs.push_back( std::make_pair( 0,3 ) );
	frs.push_back( std::make_pair( 3,6 ) );
	frs.push_back( std::make_pair( 6,9 ) );
	
	DenseSubrangeKernel kernel( kernels, frs );
	DenseScaledKernel scale( &kernel );
	
	NormalizeKernelUnitVariance<> normalizer;
	normalizer.train( scale, data );
	std::cout << "    Done training. Factor is " << scale.factor() << std::endl;
	std::cout << "    Mean                   = " << normalizer.mean() << std::endl;
	std::cout << "    Trace                  = " << normalizer.trace() << std::endl << std::endl;
	//check in feature space
	double control = 0.0;
	for ( std::size_t i=0; i<num_points; i++ ) {
		control += scale.eval(input[i], input[i]);
		for ( std::size_t j=0; j<num_points; j++ ) {
			control -= scale.eval(input[i],input[j]) / num_points;
		}
	}
	control /= num_points;
	std::cout << "    Variance of scaled MklKernel: " << control << std::endl;
}
