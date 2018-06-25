#include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/WeightedSumKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/SubrangeKernel.h>
#include <shark/Data/DataDistribution.h>
#include <shark/Core/Random.h>

using namespace shark;

//our problem
class UniformPoints : public DataDistribution<RealVector>
{
public:
	UniformPoints(std::size_t dimensions): DataDistribution<RealVector>(dimensions){}

	void draw(RealVector& input)const{
		input.resize(shape().numElements());
		for ( std::size_t j=0; j<input.size(); j++ ) {
			input(j) = random::uni(random::globalRng, -1,1);
		}
	}
};

int main()
{
	std::cout << "\n ----- Starting MklKernel normalization demo ---- \n\n" << std::flush;
	
	std::size_t num_dims = 9;
	std::size_t num_points = 200;
	UniformPoints problem(num_dims);
	Data<RealVector> data = problem.generateDataset(num_points);
	
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
	for (auto const& elem_i: data.elements()){
		control += scale.eval(elem_i, elem_i);
		for (auto const& elem_j: data.elements()){
			control -= scale.eval(elem_i, elem_j) / num_points;
		}
	}
	control /= num_points;
	std::cout << "    Variance of scaled MklKernel: " << control << std::endl;
}
