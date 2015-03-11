//===========================================================================
/*!
 * 
 *
 * \brief       Model for "soft" clustering.
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

#ifndef SHARK_MODELS_CLUSTERING_SOFTCLUSTERINGMODEL_H
#define SHARK_MODELS_CLUSTERING_SOFTCLUSTERINGMODEL_H


#include <shark/Models/Clustering/ClusteringModel.h>


namespace shark {


///
/// \brief Model for "soft" clustering.
///
/// \par
/// The SoftClusteringModel is based on an AbstractClustering
/// object. Given an input, the model outputs the cluster
/// membership function, which consists of a non-negative
/// value per cluster, and all memberships add up to one.
///
/// \par
/// See also HardClusteringModel for the best matching cluster.
///
template <class InputT>
class SoftClusteringModel : public ClusteringModel<InputT, RealVector>
{
	typedef ClusteringModel<InputT, RealVector> base_type;
	typedef AbstractClustering<InputT> ClusteringType;
	typedef typename base_type::InputType InputType;
	typedef typename base_type::OutputType OutputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	typedef typename base_type::BatchInputType BatchInputType;
public:
	/// Constructor
	SoftClusteringModel(ClusteringType* clustering)
	: base_type(clustering){
		SHARK_CHECK(
			clustering->hasSoftMembershipFunction(), 
			"[SoftClusteringModel] Clustering does not support soft membership function"
		);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SoftClusteringModel"; }

	/// \brief Compute best matching cluster.
	///
	/// \par
	/// The actual computation is redirected to the clustering object.
	void eval(InputType const & pattern, OutputType& output)const{
		output = this->mep_clustering->softMembership(pattern);
	}

	/// \brief Compute best matching cluster for a batch of inputs.
	///
	/// \par
	/// The actual computation is redirected to the clustering object.
	void eval(BatchInputType const & patterns, BatchOutputType& outputs)const{
		outputs = this->mep_clustering->softMembership(patterns);
	}
};


}
#endif
