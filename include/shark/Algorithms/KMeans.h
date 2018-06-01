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


#ifndef SHARK_ALGORITHMS_KMEANS_H
#define SHARK_ALGORITHMS_KMEANS_H

#include <shark/Core/DLLSupport.h>
#include <shark/Data/Dataset.h>
#include <shark/Models/Clustering/Centroids.h>
#include <shark/Models/RBFLayer.h>
#include <shark/Models/Kernels/KernelExpansion.h>
#include <shark/Models/Kernels/KernelHelpers.h>


namespace shark{


///
/// \brief The k-means clustering algorithm.
///
/// \par
/// The k-means algorithm takes vector-valued data
/// \f$ \{x_1, \dots, x_n\} \subset \mathbb R^d \f$
/// and splits it into k clusters, based on centroids
/// \f$ \{c_1, \dots, c_k\} \f$.
/// The result is stored in a Centroids object that can be used to
/// construct clustering models.
///
/// \par
/// This implementation starts the search with the given centroids,
/// in case the provided centroids object (third parameter) contains
/// a set of k centroids. Otherwise the search starts from the first
/// k data points.
///
/// \par
/// Note that the data set needs to include at least k data points
/// for k-means to work. This is because the current implementation
/// does not allow for empty clusters.
///
/// \param data           vector-valued data to be clustered
/// \param k              number of clusters
/// \param centroids      centroids input/output
/// \param maxIterations  maximum number of k-means iterations; 0: unlimited
/// \return               number of k-means iterations
/// \ingroup clustering
SHARK_EXPORT_SYMBOL std::size_t kMeans(Data<RealVector> const& data, std::size_t k, Centroids& centroids, std::size_t maxIterations = 0);

///
/// \brief The k-means clustering algorithm for initializing an RBF Layer
///
/// \par
/// The k-means algorithm takes vector-valued data
/// \f$ \{x_1, \dots, x_n\} \subset \mathbb R^d \f$
/// and splits it into k clusters, based on centroids
/// \f$ \{c_1, \dots, c_k\} \f$.
/// The result is stored in a RBFLayer object that can be used to
/// construct clustering models.
///
/// \par
/// This is just an alternative frontend to the version using Centroids. it creates a centroid object,
///  with as many clusters as are outputs in the RBFLayer and copies the result into the model.
///
/// \par
/// Note that the data set needs to include at least k data points
/// for k-means to work. This is because the current implementation
/// does not allow for empty clusters.
///
/// \param data           vector-valued data to be clustered
/// \param model     RBFLayer input/output
/// \param maxIterations  maximum number of k-means iterations; 0: unlimited
/// \return               number of k-means iterations
///
SHARK_EXPORT_SYMBOL std::size_t kMeans(Data<RealVector> const& data, RBFLayer& model, std::size_t maxIterations = 0);

///
/// \brief The kernel k-means clustering algorithm
///
/// \par
/// The kernel k-means algorithm takes data
/// \f$ \{x_1, \dots, x_n\} \f$
/// and splits it into k clusters, based on centroids
/// \f$ \{c_1, \dots, c_k\} \f$.
/// The centroids are elements of the reproducing kernel Hilbert space
/// (RHKS) induced by the kernel function. They are functions, represented
/// as the components of a KernelExpansion object. I.e., given a data point
/// x, the kernel expansion returns a k-dimensional vector f(x), which is
/// the evaluation of the centroid functions on x. The value of the
/// centroid function represents the inner product of the centroid with
/// the kernel-induced feature vector of x (embedding of x into the RKHS).
/// The distance of x from the centroid \f$ c_i \f$ is computes as the
/// kernel-induced distance
/// \f$ \sqrt{ kernel(x, x) + kernel(c_i, c_i) - 2 kernel(x, c_i) } \f$.
/// For the Gaussian kernel (and other normalized kernels) is simplifies to
/// \f$ \sqrt{ 2 - 2 kernel(x, c_i) } \f$. Hence, larger function values
/// indicate smaller distance to the centroid.
///
/// \par
/// Note that the data set needs to include at least k data points
/// for k-means to work. This is because the current implementation
/// does not allow for empty clusters.
///
/// \param dataset        vector-valued data to be clustered
/// \param k              number of clusters
/// \param kernel         kernel function object
/// \param maxIterations  maximum number of k-means iterations; 0: unlimited
/// \return               centroids (represented as functions, see description)
///
template<class InputType>
KernelExpansion<InputType> kMeans(Data<InputType> const& dataset, std::size_t k, AbstractKernelFunction<InputType>& kernel, std::size_t maxIterations = 0){
	if(!maxIterations)
		maxIterations = std::numeric_limits<std::size_t>::max();
	
	std::size_t n = dataset.numberOfElements();
	RealMatrix kernelMatrix = calculateRegularizedKernelMatrix(kernel,dataset,0);
	UIntVector clusterMembership(n);
	UIntVector clusterSizes(k,0);
	RealVector ckck(k,0);
	
	//init cluster assignments
	for(unsigned int i = 0; i != n; ++i){
		clusterMembership(i) = i % k;
	}
	std::shuffle(clusterMembership.begin(),clusterMembership.end(),random::globalRng);
	for(std::size_t i = 0; i != n; ++i){
		++clusterSizes(clusterMembership(i));
	}
	
	// k-means loop
	std::size_t iter = 0;
	bool equal = false;
	for(; iter != maxIterations && !equal; ++iter) {
		//the clustering model results in centers c_k= 1/n_k sum_i k(x_i,.) for all x_i points of cluster k
		//we need to compute the squared distances between all centers and points that is
		//d^2(c_k,x_i) = <c_k,c_k> -2 < c_k,x_i> + <x_i,x_i> for the i-th point.
		//thus we precompute <c_k,c_k>= sum_ij k(x_i,x_j)/(n_k)^2 for all x_i,x_j points of cluster k
		ckck.clear();
		for(std::size_t i = 0; i != n; ++i){
			std::size_t c1 = clusterMembership(i);
			for(std::size_t j = 0; j != n; ++j){
				std::size_t c2 = clusterMembership(j);
				if(c1 != c2) continue;
				ckck(c1) += kernelMatrix(i,j);
			}
		}
		noalias(ckck) = safe_div(ckck,sqr(clusterSizes),0);

		UIntVector newClusterMembership(n);
		RealVector currentDistances(k);
		for(std::size_t i = 0; i != n; ++i){
			//compute squared distances between the i-th point and the centers
			//we skip <x_i,x_i> as it is always the same for all elements and we don't need it for comparison
			noalias(currentDistances) = ckck;
			for(std::size_t j = 0; j != n; ++j){
				std::size_t c = clusterMembership(j);
				currentDistances(c) -= 2* kernelMatrix(i,j)/clusterSizes(c);
			}
			//choose the index with the smallest distance as new cluster
			newClusterMembership(i) = (unsigned int) arg_min(currentDistances);
		}
		equal = boost::equal(
			newClusterMembership,
			clusterMembership
		);
		noalias(clusterMembership) = newClusterMembership;
		//compute new sizes of clusters
		clusterSizes.clear();
		for(std::size_t i = 0; i != n; ++i){
			++clusterSizes(clusterMembership(i));
		}

		//if a cluster is empty then assign a random point to it
		for(unsigned int i = 0; i != k; ++i){
			if(clusterSizes(i) == 0){
				std::size_t elem = random::uni(random::globalRng, std::size_t(0), n-1);
				--clusterSizes(clusterMembership(elem));
				clusterMembership(elem)=i;
				clusterSizes(i) = 1;
			}
		}
	}
	
	//copy result in the expansion
	KernelExpansion<InputType> expansion;
	expansion.setStructure(&kernel,dataset,true,k);
	expansion.offset() = -ckck;
	expansion.alpha().clear();
	for(std::size_t i = 0; i != n; ++i){
		std::size_t c = clusterMembership(i);
		expansion.alpha()(i,c) = 2.0 / clusterSizes(c);
	}

	return expansion;
}

} // namespace shark
#endif
