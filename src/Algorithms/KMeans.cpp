//===========================================================================
/*!
 * 
 *
 * \brief       The k-means clustering algorithm.
 * 
 * 
 *
 * \author      T. Glasmachers
 * \date        2011
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
//===========================================================================

#define SHARK_COMPILE_DLL
#include <shark/Algorithms/KMeans.h>
#include <shark/Models/Clustering/HardClusteringModel.h>
#include <shark/Data/DataView.h>

#include <boost/range/algorithm/equal.hpp>
#include <limits>
using namespace shark;


std::size_t shark::kMeans(Data<RealVector> const& dataset, std::size_t k, Centroids& centroids, std::size_t maxIterations){
	SIZE_CHECK(k <= dataset.numberOfElements());
	if(!maxIterations)
		maxIterations = std::numeric_limits<std::size_t>::max();
	
	// initialization
	auto data = elements(dataset);
	std::size_t ell = data.size();
	std::size_t dimension = dataDimension(data);
	
	//if the centers are not already initialized, do it now
	if (centroids.numberOfClusters() != k){
		centroids.initFromData(dataset,k);
	}
	HardClusteringModel<RealVector> model(&centroids);
	auto clusterMembership = createLabeledData(dataset,model(dataset));

	// k-means loop
	std::size_t iter = 0;
	bool equal = false;
	for(; iter != maxIterations && !equal; ++iter) {
		// compute new centers
		std::vector<std::size_t> numPoints(k,0); // number of points in each cluster
		std::vector<RealVector> newCenters(k,RealVector(dimension,0.0));
			
		for(auto element: elements(clusterMembership)){
			std::size_t j = element.label;
			noalias(newCenters[j]) += element.input;
			numPoints[j]++;
		}
		for (std::size_t j=0; j<k; j++) {
			if (numPoints[j] == 0) {
				// empty cluster - assign random training point
				std::size_t index = random::discrete(random::globalRng, std::size_t(0), ell-1);
				newCenters[j] = data[index];
			}
			else {
				newCenters[j] /= (double)numPoints[j];
			}
		}
		centroids.setCentroids(createDataFromRange(newCenters));
		
		//compute new cluster memberships and check whether they are 
		// equal to the old one, in that case we stop after this iteration
		Data<unsigned int> newClusterMembership = model(dataset);
		equal = boost::equal(
			elements(newClusterMembership),
			elements(clusterMembership.labels())
		);
		clusterMembership.labels() = newClusterMembership;
	}

	// return the number of iterations
	return iter;
}

std::size_t shark::kMeans(Data<RealVector> const& data, RBFLayer& model, std::size_t maxIterations){
	//calculate clustering
	Centroids centroids;
	std::size_t iter = kMeans(data,model.outputShape().numElements(),centroids,maxIterations);
	model.centers() = createBatch<RealVector>(elements(centroids.centroids()));
	return iter;
}
