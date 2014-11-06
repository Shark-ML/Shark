//===========================================================================
/*!
 * 
 *
 * \brief       Test case for optimization of the hyperparameters of a
 * Gaussian Process/Regularization Network using evidence/marginal
 * likelihood maximization.
 * 
 * 
 *
 * \author      Christian Igel, Oswin Krause
 * \date        2011
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

#include <boost/foreach.hpp>

#include <shark/Algorithms/Trainers/PCA.h>
#include <shark/Data/Statistics.h>
#include <shark/Statistics/Distributions/MultiVariateNormalDistribution.h>

#define BOOST_TEST_MODULE ALGORITHM_PCA
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace std;
using namespace shark;

///Principal components of our multivariate data distribution, we will
///use them later for checking
double principalComponents[3][3] =
{
	{ 5, 0, 0},
	{ 0, 2, 2},
	{ 0,-1, 1}
};

///The 3D test distribution is just a multivariate Gaussian.
UnlabeledData<RealVector> createData3D()
{
	const unsigned numberOfExamples = 30000;

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
	MultiVariateNormalDistribution distribution(covariance);

	//and we sample from it
	std::vector<RealVector> data(numberOfExamples);
	BOOST_FOREACH(RealVector& sample, data)
	{
		//first element is the sample, second is the underlying uniform gaussian
		sample = mean + distribution().first;
	}
	return  createDataFromRange(data);
}

///The 2D test distribution is an even simpler Gaussian.
UnlabeledData<RealVector> createData2D()
{
	const unsigned numberOfExamples = 10000;

	RealMatrix C(2, 2);
	RealVector mu(2);
	C.clear();
	C(0, 0) = 16.;
	C(1, 1) = 1.;
	mu(0) = mu(1) = 1.;
	MultiVariateNormalDistribution distribution(C);

	std::vector<RealVector> v;
	for(unsigned i=0; i<numberOfExamples; i++) v.push_back(mu + distribution().first);
	return  createDataFromRange(v);
}


///The test distribution here is the same as in createData3D, but in this case
///we add a lot of low variance uncorrelatd variables on top. after that w generate a very small
///dataset, such that the simple PCA computation does not work anymore.
UnlabeledData<RealVector> createDataNotFullRank()
{
	const unsigned dimensions = 10;
	const unsigned numberOfExamples = 5;

	RealVector mean(dimensions,0);
	mean(0) = 1;
	mean(1) = -1;
	mean(2) = 3;

	// to create the covariance matrix we first put the
	// copy the principal components  in the matrix
	// and than use an outer product
	RealMatrix covariance(dimensions,dimensions,0.0);
	diag(covariance) = blas::repeat(0.001,dimensions);
	for(int i = 0; i != 3; ++i)
	{
		for(int j = 0; j != 3; ++j)
		{
			covariance(i,j) = principalComponents[i][j];
		}
	}
	covariance = prod(trans(covariance),covariance);

	//now we can create the distribution
	MultiVariateNormalDistribution distribution(covariance);

	//and we sample from it
	std::vector<RealVector> data(numberOfExamples);
	BOOST_FOREACH(RealVector& sample, data)
	{
		//first element is the sample, second is the underlying uniform gaussian
		sample = mean + distribution().first;
	}
	return  createDataFromRange(data,2);//small batch size to get batching errors
}

BOOST_AUTO_TEST_SUITE (Algorithms_Trainers_PCA)

BOOST_AUTO_TEST_CASE( PCA_TEST_MORE_DATA_THAN_DIMENSIONS ){
	//
	// 1. 2D test with whitening and without dimensionality
	// reduction
	//

	// create 2D sample data
	UnlabeledData<RealVector> data = createData2D();
	Data<RealVector> encodedData, decodedData;
	
	// compute statistics
	RealVector mean, var;
	meanvar(data, mean, var);
	cout << "data mean\t" << mean << "  \tvariance\t" << var  << endl;

	// do PCA with whitening
	bool whitening = true;
	PCA pca(data, whitening);

	// encode data and compute statistics
	LinearModel<> enc;
	pca.encoder(enc);
	encodedData = enc(data);
	RealVector emean, evar;
	meanvar(encodedData, emean, evar);
	cout << "encoded mean\t" << emean << " \tvariance\t" << evar  << endl;

	// decode data again  and compute statistics
	LinearModel<> dec;
	pca.decoder(dec);
	RealVector dmean, dvar;
	decodedData = dec(encodedData);
	meanvar(decodedData, dmean, dvar);
	cout << "decoded mean\t" << dmean << "  \tvariance\t" << dvar  << endl;

	/// do checks
	for(unsigned i=0; i<2; i++) {
		// have mean and variance correctly been reconstructed
		BOOST_CHECK_SMALL(mean(i) - dmean(i), 1.e-6);
		BOOST_CHECK_SMALL(var(i) - dvar(i), 1.e-6);
		// is the variance one after whitening
		BOOST_CHECK_SMALL(evar(i) - 1., 1.e-10);
		// is the mean zero after PCA
		BOOST_CHECK_SMALL(emean(i), 1.e-10);
	}

	// 
	// 3D test case with dimensionalty reduction, without
	// whitening
	//
	data = createData3D();

	// With the definition of the model, we declare, how many
	// principal components we want.  If we want all, we set
	// inputs=outputs = 3 but since want to do a reduction, we use
	// only 2 in the second argument.  The third argument is the
	// bias. The PCA class needs a bias to work.
	LinearModel<> pcaModel(3,2,true);
	pca.setWhitening(false);
	pca.train(pcaModel,data);

	RealVector pc1 = row(pcaModel.matrix(),0) * sqrt(pca.eigenvalues()(0));
	RealVector pc2 = row(pcaModel.matrix(),1) * sqrt(pca.eigenvalues()(1));	
	cout << "principal component 1: " << pc1 << endl;
	cout << "principal component 2: " << pc2 << endl;

	// Check whether results are close to be expected form the
	// covariance matrix of the underlying distribution. Because
	// of samplig (and also numerical errors), the tolerance is
	// pretty high.
	for(unsigned i=0; i<3; i++) {
		BOOST_CHECK_SMALL(fabs(principalComponents[0][i]) - fabs(pc1(i)), 0.05);
		BOOST_CHECK_SMALL(fabs(principalComponents[1][i]) - fabs(pc2(i)), 0.05);
	}
}


BOOST_AUTO_TEST_CASE( PCA_TEST_LESS_DATA_THAN_DIMENSIONS ){

	UnlabeledData<RealVector> data = createDataNotFullRank();
	Data<RealVector> encodedData, decodedData;
	// compute statistics
	RealVector mean, var;
	meanvar(data, mean, var);
	
	// do PCA with whitening
	bool whitening = true;
	PCA pca(data, whitening);

	// encode data and compute statistics
	LinearModel<> enc;
	pca.encoder(enc);
	encodedData = enc(data);
	RealVector emean;
	RealMatrix ecovar;
	meanvar(encodedData, emean, ecovar);

	// decode data again  and compute statistics
	LinearModel<> dec;
	pca.decoder(dec);
	RealVector dmean, dvar;
	decodedData = dec(encodedData);
	meanvar(decodedData, dmean, dvar);

	/// do checks
	for(unsigned i=0; i<mean.size(); i++) {
		// have mean and variance correctly been reconstructed
		BOOST_CHECK_SMALL(mean(i) - dmean(i), 1.e-5);
		BOOST_CHECK_SMALL(var(i) - dvar(i), 1.e-4);
	}
	
	for(unsigned i=0; i<emean.size()-1; i++) {
		for(std::size_t j = 0; j < i; ++j){
			//covariance must be 0
			BOOST_CHECK_SMALL(ecovar(i,j) , 1.e-8);
		}			
		// is the variance 1 after whitening
		BOOST_CHECK_SMALL(ecovar(i,i) - 1., 1.e-8);
		// is the mean zero after PCA
		BOOST_CHECK_SMALL(emean(i), 1.e-9);
	}
}

BOOST_AUTO_TEST_SUITE_END()
