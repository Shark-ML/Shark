//===========================================================================
/*!
 * 
 * \file        ClusteringModel.h
 *
 * \brief       Super class for clustering models.
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

#ifndef SHARK_MODELS_CLUSTERING_CLUSTERINGMODEL_H
#define SHARK_MODELS_CLUSTERING_CLUSTERINGMODEL_H


#include <shark/Models/Clustering/AbstractClustering.h>
#include <shark/Models/AbstractModel.h>


namespace shark {


/// \brief Abstract model with associated clustering object.
///
/// See HardClusteringModel and SoftClusteringModel for details.
template <class InputT, class OutputT>
class ClusteringModel : public AbstractModel<InputT, OutputT>
{
public:
	typedef AbstractModel<InputT, OutputT> base_type;
	typedef AbstractClustering<InputT> ClusteringType;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// Constructor.
	ClusteringModel(ClusteringType* clustering)
	: mep_clustering(clustering)
	{ SHARK_CHECK(clustering, "[ClusteringModel] Clustering must not be NULL"); }


	/// Redirect parameter access to the clustering object
	RealVector parameterVector() const
	{ return mep_clustering->parameterVector(); }

	/// Redirect parameter access to the clustering object
	void setParameterVector(RealVector const& newParameters)
	{ mep_clustering->setParameterVector(newParameters); }

	/// Redirect parameter access to the clustering object
	std::size_t numberOfParameters() const
	{ return mep_clustering->numberOfParameters(); }

	/// From ISerializable, reads a model from an archive.
	void read(InArchive& archive)
	{ archive & *mep_clustering; }

	/// From ISerializable, writes a model to an archive.
	void write(OutArchive& archive) const
	{ archive & *mep_clustering; }

	using base_type::eval;
	void eval(BatchInputType const& patterns, BatchOutputType& outputs,  State& state)const{
		eval(patterns,outputs);
	}

protected:
	/// Clustering object, see class AbstractClustering
    ClusteringType* mep_clustering;
};


}
#endif
