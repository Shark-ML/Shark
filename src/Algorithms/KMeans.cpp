//===========================================================================
/*!
 *  \brief The k-means clustering algorithm.
 *
 *  \author T. Glasmachers
 *  \date 2011
 *
 *  \par Copyright (c) 1998-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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


#include <shark/Algorithms/KMeans.h>
#include <shark/Models/Clustering/HardClusteringModel.h>
#include <shark/Data/DataView.h>

#include <boost/range/algorithm/equal.hpp>
using namespace shark;


std::size_t shark::kMeans(Data<RealVector> const& dataset, std::size_t k, Centroids& centroids, std::size_t maxIterations){
	SIZE_CHECK(k <= dataset.numberOfElements());
	
	// initialization
	std::size_t ell = dataset.numberOfElements();
	std::size_t dimension = dataDimension(dataset);
	
	//if the centers are not already initialized, do it now
	if (centroids.numberOfClusters() != k){
		centroids.initFromData(dataset,k);
	}
	HardClusteringModel<RealVector> model(&centroids);
	ClassificationDataset clustersAssignmentSet(dataset,model(dataset));
	DataView<ClassificationDataset> clusters(clustersAssignmentSet);
	
	// k-means loop
	std::size_t iter = 0;
	for(; !maxIterations || iter != maxIterations; ++iter) {
		//we don't need to evaluate in the first iteration
		if(iter != 0){
			Data<unsigned int> newClusterAssignment = model(dataset);
			//if the new clustering is the same, stop 
			bool equal = boost::equal(
				newClusterAssignment.elements(),
				clustersAssignmentSet.labels().elements()
			);
			if(equal)
				break;
			swap(newClusterAssignment,clustersAssignmentSet.labels());
		}
		// compute new centers
		std::vector<std::size_t> numPoints(k,0); // number of points in each cluster
		std::vector<RealVector> newCenters(k,RealVector(dimension,0.0));
			
		for(std::size_t i = 0; i != ell; ++i) {
			std::size_t j = clusters[i].label;
			noalias(newCenters[j]) += clusters[i].input;
			numPoints[j]++;
		}
		for (std::size_t j=0; j<k; j++) {
			if (numPoints[j] == 0) {
				// empty cluster - assign random training point
				newCenters[j] = clusters[Rng::discrete(0, ell-1)].input;
			}
			else {
				newCenters[j] /= (double)numPoints[j];
			}
		}
		centroids.setCentroids(createDataFromRange(newCenters));
	}

	// return the number of iterations
	return iter;
}
