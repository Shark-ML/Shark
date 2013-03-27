//===========================================================================
/*!
*
*  \brief Model for "soft" clustering.
*
*  \author  T. Glasmachers
*  \date    2011
*
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
