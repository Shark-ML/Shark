//===========================================================================
/*!
*  \brief Test case for k-means clustering.
*
*  \author  T. Glasmachers
*  \date    2011
*
*  \par Copyright (c) 2011:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*
*
*  <BR><HR>
*  This file is part of Shark. This library is free software;
*  you can redistribute it and/or modify it under the terms of the
*  GNU General Public License as published by the Free Software
*  Foundation; either version 3, or (at your option) any later version.
*
*  This library is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this library; if not, see <http://www.gnu.org/licenses/>.
*  
*/
//===========================================================================

#define BOOST_TEST_MODULE Algorithms_KMeans
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <algorithm>

#include <shark/Algorithms/KMeans.h>
#include <shark/Models/Clustering/HardClusteringModel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Converter.h>

using namespace shark;


BOOST_AUTO_TEST_CASE(KMeans_simple)
{
	RealVector v(1);

	// prepare data set
	std::vector<RealVector> data(300);
	for (std::size_t i=0; i<100; i++)
	{
		v(0) = Rng::uni();
		data[i] = v;
		v(0) = Rng::uni() + 10.0;
		data[100+i] = v;
		v(0) = Rng::uni() + 20.0;
		data[200+i] = v;
	}
	Data<RealVector> dataset = createDataFromRange(data);

	// prepare initial centroids
	std::vector<RealVector> start(3);
	v(0) =  2.0; start[0] = v;
	v(0) =  7.0; start[1] = v;
	v(0) = 25.0; start[2] = v;
	Centroids centroids( createDataFromRange(start));

	// invoke k-means
	std::size_t iterations = kMeans(dataset, 3, centroids);
	std::cout<<iterations<<std::endl;

	// check result
	Data<RealVector> const& c = centroids.centroids();
	std::cout<<c<<std::endl;
	BOOST_CHECK_EQUAL(c.numberOfElements(), 3u);
	BOOST_CHECK(c.element(0)(0) >  0.0);
	BOOST_CHECK(c.element(0)(0) <  1.0);
	BOOST_CHECK(c.element(1)(0) > 10.0);
	BOOST_CHECK(c.element(1)(0) < 11.0);
	BOOST_CHECK(c.element(2)(0) > 20.0);
	BOOST_CHECK(c.element(2)(0) < 21.0);
	BOOST_CHECK_LE(iterations, 3u);
}

// tests whether the algorithm leads to clusters which are constant (i.e. the algorithm converged to
// a stationary solution).
BOOST_AUTO_TEST_CASE(KMeans_multiple_gauss)
{
	const unsigned int numTrials = 100;
	const unsigned int numPoints = 300;
	const unsigned int numMeans = 3;
	const unsigned int numDimensions = 5;
	for(unsigned int trial = 0; trial != numTrials; ++trial){
		//prepare means
		std::vector<RealVector> means(numMeans,RealVector(numDimensions));
		for (unsigned int i=0; i<numMeans; i++){
			for (unsigned int j=0; j <numDimensions; j++){
				means[i](j) = Rng::uni(0,5);
			}
		}
		// prepare data set
		std::vector<RealVector> data(numPoints);
		for (std::size_t i=0; i<numPoints; i++)
		{
			data[i]=means[i%numMeans];
			for (unsigned int j=0; j <numDimensions; j++){
				data[i](j) += Rng::uni(0,1);
			}
		}
		Data<RealVector> dataset = createDataFromRange(data);

		// invoke k-means with random centroids and no upper limit for iterations
		Centroids centroids;
		kMeans(dataset, numMeans, centroids);

		// check result
		BOOST_REQUIRE_EQUAL( centroids.centroids().numberOfElements(), numMeans);
		BOOST_REQUIRE_EQUAL( dataDimension(centroids.centroids()), numDimensions);
		HardClusteringModel<RealVector> model(&centroids);
		
		//assign centers
		Data<unsigned int> clusters = model(dataset);
		
		//create cluster means (they should be the same as the previous ones)
		std::vector<RealVector> clusterMeans(numMeans,RealVector(numDimensions,0));
		std::vector<std::size_t> members = classSizes(clusters);
		for (std::size_t i=0; i<numPoints; i++)
		{
			unsigned int id = clusters.element(i);
			clusterMeans[id]+=data[i]/members[id];
		}
		
		//check that the means are the same
		for (unsigned int i=0; i<numMeans; i++){
			double distance = distanceSqr(clusterMeans[i],centroids.centroids().element(i));
			BOOST_CHECK_SMALL(distance, 1.e-10);
		}
	}
}


BOOST_AUTO_TEST_CASE(Kernel_KMeans_multiple_gauss)
{
	const unsigned int numTrials = 100;
	const unsigned int numPoints = 300;
	const unsigned int numMeans = 3;
	const unsigned int numDimensions = 5;
	for(unsigned int trial = 0; trial != numTrials; ++trial){
		//prepare means
		std::vector<RealVector> means(numMeans,RealVector(numDimensions));
		for (unsigned int i=0; i<numMeans; i++){
			for (unsigned int j=0; j <numDimensions; j++){
				means[i](j) = Rng::uni(0,5);
			}
		}
		// prepare data set
		std::vector<RealVector> data(numPoints);
		for (std::size_t i=0; i<numPoints; i++)
		{
			data[i]=means[i%numMeans];
			for (unsigned int j=0; j <numDimensions; j++){
				data[i](j) += Rng::uni(0,1);
			}
		}
		Data<RealVector> dataset = createDataFromRange(data);

		// invoke k-means with random centroids and no upper limit for iterations
		Centroids centroids;
		LinearKernel<> kernel;
		KernelExpansion<RealVector> clusteringModel = kMeans(dataset, numMeans, kernel);
		//std::cout<<clusteringModel.alpha()<<std::endl;
		// check result
		BOOST_REQUIRE_EQUAL( clusteringModel.outputSize(), numMeans);
		
		//check that every point is assigned to exactly one cluster
		UIntVector numAssignments = sum_columns(clusteringModel.alpha() != 0.0);
		BOOST_REQUIRE_EQUAL(numAssignments.size(),numPoints);
		for(std::size_t i = 0; i != numPoints; ++i){
			BOOST_CHECK_EQUAL(numAssignments(i),1);
		}
		//check that the weights of every cluster sums to ~1
		RealVector columnSums = sum_rows(clusteringModel.alpha());
		BOOST_REQUIRE_EQUAL(columnSums.size(),numMeans);
		for(std::size_t i = 0; i != numMeans; ++i){
			BOOST_CHECK_CLOSE(columnSums(i),2.0, 1.e-5);
		}
		
		//now calculate the exact centers and check that the distances are 
		//the same as the result of the kernel expansion
		RealMatrix centers(numMeans,numDimensions,0.0);
		for(std::size_t i = 0; i != numPoints; ++i){
			centers+=0.5*outer_prod(row(clusteringModel.alpha(),i),data[i]);
		}
		
		for(std::size_t b = 0; b != dataset.numberOfBatches(); ++b){
			RealMatrix const& batch = dataset.batch(b);
			RealMatrix modelResult = clusteringModel(batch);
			RealMatrix kernelResult = -kernel.featureDistanceSqr(batch,centers);
			BOOST_REQUIRE_EQUAL(modelResult.size1(),kernelResult.size1());
			BOOST_REQUIRE_EQUAL(modelResult.size2(),kernelResult.size2());
			//the distances are off by the size of the elements so we have to correct this 
			//(it does not matter for clustering)
			for(std::size_t i = 0; i != batch.size1();++i){
				double distSqri = norm_sqr(row(batch,i));
				row(modelResult,i) -= blas::repeat(distSqri,numMeans);
				BOOST_CHECK_SMALL(distanceSqr(row(modelResult,i),row(kernelResult,i)),1.e-5);
			}
		}
		
		//now check proper convergence of the algorithms, that means that cluster assignment
		//is stable.
		ArgMaxConverter<KernelExpansion<RealVector> > model;
		model.decisionFunction() = clusteringModel;
		//assign centers
		Data<unsigned int> clusters = model(dataset);
		
		//create new cluster means (they should be the same as the previous ones)
		std::vector<RealVector> newClusterMeans(numMeans,RealVector(numDimensions,0));
		std::vector<std::size_t> members = classSizes(clusters);
		for (std::size_t i=0; i<numPoints; i++)
		{
			unsigned int id = clusters.element(i);
			newClusterMeans[id]+=data[i]/members[id];
		}
		
		//check that the means are the same
		for (unsigned int i=0; i<numMeans; i++){
			double distance = distanceSqr(newClusterMeans[i],row(centers,i));
			BOOST_CHECK_SMALL(distance, 1.e-10);
		}
	}
}
