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
 * \par Copyright 1995-2015 Shark Development Team
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
//===========================================================================

#ifndef SHARK_MODELS_CLUSTERING_CENTROIDS_H
#define SHARK_MODELS_CLUSTERING_CENTROIDS_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/Clustering/AbstractClustering.h>
#include <shark/Data/Dataset.h>


namespace shark {


/// \brief Clusters defined by centroids.
///
/// \par
/// Centroids are an elementary way to define clusters by means
/// of the one-nearest-neighbor rule. This rule defines a hard
/// clustering decision.
///
/// \par
/// The Centroids class uses inverse distances to compute soft
/// clustering memberships. This is arbitrary and can be changed
/// by overriding the membershipKernel function.
///
class Centroids : public AbstractClustering<RealVector>
{
	typedef AbstractClustering<RealVector> base_type;

public:
	/// Default constructor
	SHARK_EXPORT_SYMBOL Centroids();

	/// Constructor
	///
	/// \param  centroids  number of centroids in the model (initially zero)
	/// \param  dimension  dimension of the input space, and thus of the centroids
	SHARK_EXPORT_SYMBOL Centroids(std::size_t centroids, std::size_t dimension);

	/// Constructor
	///
	/// \param  centroids  centroid vectors
	SHARK_EXPORT_SYMBOL Centroids(Data<RealVector> const& centroids);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Centroids"; }

	/// from IParameterizable
	SHARK_EXPORT_SYMBOL RealVector parameterVector() const;

	/// from IParameterizable
	SHARK_EXPORT_SYMBOL void setParameterVector(RealVector const& newParameters);

	/// from IParameterizable
	SHARK_EXPORT_SYMBOL std::size_t numberOfParameters() const;

	/// return the dimension of the inputs
	std::size_t dimension() const
	{
		return dataDimension(m_centroids);
	}

	/// return the number of centroids in the model
	SHARK_EXPORT_SYMBOL std::size_t numberOfClusters() const;

	/// read access to the centroid vectors
	Data<RealVector> const& centroids() const{
		return m_centroids;
	}

	/// overwrite the centroid vectors
	void setCentroids(Data<RealVector> const& newCentroids){
		m_centroids = newCentroids;
	}

	/// from ISerializable
	SHARK_EXPORT_SYMBOL void read(InArchive& archive);

	/// from ISerializable
	SHARK_EXPORT_SYMBOL void write(OutArchive& archive) const;

	/// from AbstractClustering: Compute cluster memberships.
	SHARK_EXPORT_SYMBOL RealVector softMembership(RealVector const& pattern) const;
	/// From AbstractClustering: Compute cluster memberships for a batch of patterns.
	SHARK_EXPORT_SYMBOL RealMatrix softMembership(BatchInputType const& patterns) const;
	
	/// Computes the distances of each pattern to all cluster centers
	SHARK_EXPORT_SYMBOL RealMatrix distances(BatchInputType const& patterns) const;


	/// initialize centroids from labeled data: take the first
	/// data points with different labels; if there are more
	/// centroids than classes, the remaining centroids are filled
	/// with the first elements in the data set
	///
	/// \param  data  dataset from which to take the centroids
	/// \param  noClusters  number of centroids in the model, default 0 is mapped to the number of classes in the data set
	/// \param  noClasses  number of clases in the dataset, default 0 means that the number is computed 
	SHARK_EXPORT_SYMBOL void initFromData(ClassificationDataset const& data, unsigned noClusters = 0, unsigned noClasses = 0);

	/// initialize centroids from unlabeled data: 
	/// take a random subset of data points
	///
	/// \param  dataset dataset from which to take the centroids
	/// \param  noClusters  number of centroids in the model
	SHARK_EXPORT_SYMBOL void initFromData(Data<RealVector> const& dataset, unsigned noClusters);

protected:
	/// Compute unnormalized membership from distance.
	/// The default implementation is to return exp(-distance)
	SHARK_EXPORT_SYMBOL virtual double membershipKernel(double dist) const;

	/// centroid vectors
	Data<RealVector> m_centroids;
};


}
#endif
