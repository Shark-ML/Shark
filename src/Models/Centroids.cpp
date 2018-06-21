//===========================================================================
/*!
 * 
 *
 * \brief       Clusters defined by centroids.
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
#include <shark/Models/Clustering/Centroids.h>
#include <shark/Data/DataView.h>

using namespace shark;


Centroids::Centroids(){
	this->m_features |= HAS_SOFT_MEMBERSHIP;
}


Centroids::Centroids(std::size_t centroids, std::size_t dim)
: m_centroids(centroids, dim){
	this->m_features |= HAS_SOFT_MEMBERSHIP;
}

Centroids::Centroids(Data<RealVector> const& centroids)
: m_centroids(centroids)
{
	this->m_features |= HAS_SOFT_MEMBERSHIP;
}


RealVector Centroids::parameterVector() const{
	RealVector param(numberOfParameters());
	std::size_t pos = 0;
	for(auto const& mat:m_centroids.batches()){
		auto vec = to_vector(mat);
		noalias(subrange(param,pos,pos+vec.size())) = vec;
		pos += vec.size();
	}
	return param;
}

void Centroids::setParameterVector(RealVector const& newParameters){
	std::size_t pos = 0;
	for(auto& mat:m_centroids.batches()){
		auto vec = to_vector(mat);
		noalias(vec) = subrange(newParameters,pos,pos+vec.size());
		pos += vec.size();
	}
}

std::size_t Centroids::numberOfParameters() const{
	std::size_t centroids = numberOfClusters();
	return (centroids == 0 ? 0 : dataDimension(m_centroids) * centroids);
}

std::size_t Centroids::numberOfClusters() const{
	return m_centroids.numberOfElements();
}

void Centroids::read(InArchive& archive){
	archive & m_centroids;
}

void Centroids::write(OutArchive& archive) const{
	archive & m_centroids;
}

RealMatrix Centroids::distances(BatchInputType const& patterns) const{
	std::size_t numClusters = numberOfClusters();
	std::size_t numPatterns = patterns.size1();
	RealMatrix distances(numPatterns, numClusters);
	//first evaluate distance to all centroids;
	std::size_t batchBegin = 0;
	for (std::size_t i=0; i != m_centroids.numberOfBatches(); i++){
		std::size_t batchEnd = batchBegin + m_centroids.batch(i).size1();
		columns(distances,batchBegin,batchEnd) = sqrt(distanceSqr(patterns, m_centroids.batch(i)));
		batchBegin = batchEnd;
	}
	return distances;
}

RealVector Centroids::softMembership(RealVector const& pattern) const{
	std::size_t numClusters = numberOfClusters();
	RealVector membership(numClusters);
	//first evaluate distance to all centroids;
	std::size_t batchBegin = 0;
	for (std::size_t i=0; i != m_centroids.numberOfBatches(); i++){
		std::size_t batchEnd = batchBegin + m_centroids.batch(i).size1();
		subrange(membership,batchBegin,batchEnd) = sqrt(distanceSqr(pattern, m_centroids.batch(i)));
		batchBegin = batchEnd;
	}
	//apply membership kernels and normalize to 1
	for (std::size_t i=0; i != numClusters; i++){
		membership(i) = membershipKernel(membership(i));
	}
	membership /= sum(membership);
	return membership;
}

RealMatrix Centroids::softMembership(BatchInputType const& patterns) const{
	std::size_t numClusters = numberOfClusters();
	std::size_t numPatterns = patterns.size1();
	RealMatrix membership = distances(patterns);
	//apply membership kernels and normalize to 1
	for (std::size_t i=0; i != numPatterns; i++){
		for (std::size_t j=0; j != numClusters; j++)
			membership(i,j) = membershipKernel(membership(i,j));
		row(membership,i) /= sum(row(membership,i));
	}
	return membership;
}

double Centroids::membershipKernel( double dist ) const{
	return (dist < 1e-100 ? 1e100 : 1.0 / dist);
}

void Centroids::initFromData(const ClassificationDataset &data, std::size_t noClusters, std::size_t noClasses) {
	if(!noClasses) noClasses = shark::numberOfClasses(data); // default: recompute number of classes
	if(!noClusters) noClusters = noClasses;  // default: as many centroids as classes

	/// rule: take the first data points with different labels; if there
	/// are more centroids than classes, the remaining centroids
	/// are filled with the first elements in the data set
	std::vector< RealVector >centers;
	UIntVector flag(noClasses,0);
	std::size_t elementCount = 0; // number of centroids found so far, equal to tmp.size()
	std::size_t classCount = 0; // number of different classes encountered so far

	typedef ClassificationDataset::const_element_range Elements;
	Elements elements = data.elements();
	for(Elements::iterator it = elements.begin(); it != elements.end(); ++it) {
		// we take the element if it has a so far unseen class
		// or if the current number of centroids plus one
		// element from each class that has not been
		// encountered so far is smaller than the desired
		// number of centroids
		if((flag(it->label) == 0) || ((elementCount + noClasses - classCount) < noClusters)) {
			if(flag(it->label) == 0) {
				flag(it->label) = 1;
				classCount++;
			}
                        centers.push_back(it->input);
                        elementCount++;
                }
                if(elementCount == noClusters) break; 
	}
	setCentroids(createDataFromRange(centers));
}

void Centroids::initFromData(Data<RealVector> const& dataset, std::size_t noClusters) {
	setCentroids(toDataset(randomSubset(toView(dataset),noClusters)));
}
