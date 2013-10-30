#include <shark/Algorithms/Trainers/PCA.h>

//header needed for data generation
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>
#include <boost/foreach.hpp>//just for beauty :)

using namespace shark;
using namespace std;

///In this test, we will use PCA to calculate the
///eigenvectors of a scatter matrix and do a
///reduction of the subspace to the space
///spanned by the two eigenvectors with the biggest
///eigenvalues.

///the principal components of our multivariate data distribution
///we will use them later for checking
double principalComponents[3][3] =
{
	{ 5, 0, 0},
	{ 0, 2, 2},
	{ 0,-1, 1}
};

std::size_t numberOfExamples = 30000;

///The test distribution is just a multivariate Gaussian.
UnlabeledData<RealVector> createData()
{
	RealVector mean(3);
	mean(0) = 1;
	mean(1) = -1;
	mean(2) = 3;

	// to create the covariance matrix we first put the
	// copy the principal components  in the matrix
	// and than use an outer product
	RealMatrix covariance(3,3);
	for(int i = 0; i != 3; ++i)
	{
		for(int j = 0; j != 3; ++j)
		{
			covariance(i,j) = principalComponents[i][j];
		}
	}
	covariance = prod(trans(covariance),covariance);

	//now we can create the distribution
	MultiVariateNormalDistribution distribution(3);
	distribution.setCovarianceMatrix(covariance);

	//and we sample from it
	std::vector<RealVector> data(numberOfExamples);
	BOOST_FOREACH(RealVector& sample, data)
	{
		//first element is the sample, second is the underlying uniform gaussian
		sample = mean + distribution().first;
	}
	return createDataFromRange(data);
}


int main(){

	// We first create our problem. Since the PCA is a unsupervised Method,
	// We use UnlabeledData instead of Datas. 
	UnlabeledData<RealVector> data = createData();

	// With the definition of the model, we declare, how many
	// principal components we want.  If we want all, we set
	// inputs=outputs = 3, but since want to do a reduction, we
	// use only 2 in the second argument.  The third argument is
	// the bias. pca needs a bias to work.
	LinearModel<> pcaModel(3,2,true);

	// Now we can construct the PCA.
	// We can decide whether we want a whitened output or not.
	// For testing purposes, we don't want whitening in this
	// example.
	PCA pca;
	pca.setWhitening(false);
	pca.train(pcaModel,data);



	//Print the rescaled results.
	//Should be the same as principalComponents, except for sign change
	//and numerical errors.
	cout << "RESULTS: " << std::endl;
	cout << "======== " << std::endl << std::endl;
	cout << "principal component 1: " << row(pcaModel.matrix(),0) * sqrt(pca.eigenvalues()(0)) << std::endl;
	cout << "principal component 2: " << row(pcaModel.matrix(),1) * sqrt( pca.eigenvalues()(1) ) << std::endl;

}
