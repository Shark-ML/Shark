//===========================================================================
/*!
 * 
 * \file        AbstractClustering.h
 *
 * \brief       Super class for clustering definitions.
 * 
 * 
 *
 * \author      T. Glasmachers
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
//===========================================================================

#ifndef SHARK_MODELS_CLUSTERING_ABSTRACTCLUSTERING_H
#define SHARK_MODELS_CLUSTERING_ABSTRACTCLUSTERING_H


#include <shark/Data/BatchInterface.h>
#include <shark/Core/Flags.h>
#include <shark/Core/INameable.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Core/IConfigurable.h>
#include <shark/Core/ISerializable.h>


namespace shark {


///
/// \brief Base class for clustering.
///
/// \par
/// Clustering algorithms vary widely in the data structures
/// on which they operate. For example, simple centroid-based
/// approaches such as k-means are mutually incompatible with
/// tree-based hierarchical clustering. This interface class
/// is the attempt to cast different cluster descriptions into
/// a minimal common interface that is useful for prediction.
///
/// \par
/// There are at least two common types of predictions made
/// with clusterings. The first one is the assignment of the
/// best matching cluster to a patters, such as in vector
/// quantization or unsupervised clustering. This is often
/// referred to as "hard clustering". The second one is the
/// computation of a membership function ("soft clustering").
/// The membership of a pattern to a cluster is non-negative,
/// and the total membership should add to one.
///
/// \par
/// This interface makes minimal assumptions to allow for
/// these types of predictions. It assumes that clusters can
/// be enumbered (identified by an index), and that at least
/// a membership test can be made (hard clustering). It is
/// optional to provide a membership function. Only one of
/// the two interfaces for best matching cluster or membership
/// function need to be implemented, since the best matching
/// cluster can be deduced from the membership function.
/// However, often the best matching cluster can be computed
/// more efficiently than the membership function. In these
/// cases both interface functions should be implemented.
///
/// \par
/// The purpose of this interface is to act as a common
/// super class to different data structures describing the
/// outcome of a clustering operation. The interface allows
/// all of these data structures to be used in the two
/// clustering model classes: HardClusteringModel and
/// SoftClusteringModel.
///
template <class InputT>
class AbstractClustering : public INameable, public IParameterizable, public ISerializable, public IConfigurable
{
public:
	typedef InputT InputType;
	typedef unsigned int OutputType;
	typedef typename Batch<InputType>::type BatchInputType;
	typedef Batch<OutputType>::type BatchOutputType;

	enum Feature {
		HAS_SOFT_MEMBERSHIP = 1,
	};
	SHARK_FEATURE_INTERFACE;

	/// Tests whether the clustering can compute a (soft)
	/// member ship function, describing the membership
	/// of a sample to the different clusters.
	bool hasSoftMembershipFunction()const{
		return m_features & HAS_SOFT_MEMBERSHIP;
	}

	/// return the number of clusters
	virtual std::size_t numberOfClusters() const = 0;

	/// \brief Compute best matching cluster.
	///
	/// \par
	/// This function should be overriden by sub-classes to
	/// compute the cluster best matching the input pattern.
	/// The (typically slow) default implementation is to
	/// create a batch of size 1 and return the result of the batch call to hardMembership
	virtual unsigned int hardMembership(InputType const& pattern) const{
		typename Batch<InputType>::type b = Batch<InputType>::createBatch(pattern);
		get(b,0) = pattern;
		return hardMembership(b)(0);
	}
	
	/// \brief Compute best matching cluster for a batch of inputs.
	///
	/// \par
	/// This function should be overriden by sub-classes to
	/// compute the cluster best matching the input pattern.
	/// The (typically slow) default implementation is to
	/// return the arg max of the soft membership function for every pattern.
	virtual BatchOutputType hardMembership(BatchInputType const& patterns) const{
		std::size_t numPatterns = boost::size(patterns);
		RealMatrix f = softMembership(patterns);
		SHARK_ASSERT(f.size2() > 0);
		SHARK_ASSERT(f.size1() == numPatterns);
		BatchOutputType outputs(numPatterns);
		for(std::size_t i = 0; i != numPatterns;++i){
			RealMatrixRow membership(f,i);
			outputs(i) = std::max_element(membership.begin(),membership.end())-membership.begin();
		}
		return outputs;
	}

	/// \brief Compute cluster membership function.
	///
	/// \par
	/// This function should be overriden by sub-classes to
	/// compute the membership of a pattern to the clusters.
	/// The default implementation creates a batch of size 1 
	/// and calls the batched version. If this is not overriden, an xception is thrown.
	virtual RealVector softMembership(InputType const& pattern) const{
		return row(softMembership(Batch<InputType>::createBatch(pattern)),0);
	}
	
	/// \brief Compute cluster membership function.
	///
	/// \par
	/// This function should be overriden by sub-classes to
	/// compute the membership of a pattern to the clusters.
	/// This default implementation throws an exception.
	virtual RealMatrix softMembership(BatchInputType const& patterns) const{
		SHARK_FEATURE_EXCEPTION(HAS_SOFT_MEMBERSHIP);
	}


	/// empty default implementation of IConfigurable::configure
	virtual void configure(PropertyTree const& node) {}

	/// empty default implementation of ISerializable::read
	void read(InArchive& archive) {}

	/// empty default implementation of ISerializable::write
	void write(OutArchive& archive) const {}
};


}
#endif
