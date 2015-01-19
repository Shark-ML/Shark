//===========================================================================
/*!
 * 
 *
 * \brief       Hierarchical Clustering.
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

#ifndef SHARK_MODELS_CLUSTERING_HIERARCHICALCLUSTERING_H
#define SHARK_MODELS_CLUSTERING_HIERARCHICALCLUSTERING_H


#include <shark/Models/Clustering/AbstractClustering.h>
#include <shark/Models/Trees/BinaryTree.h>


namespace shark {


///
/// \brief Clusters defined by a binary space partitioning tree.
///
/// \par
/// Binary space-partitioning is a simple and fast way of
/// clustering.
///
/// \par
/// It is not clear how the unfolding of the tree can be
/// expressed as a flat parameter vector. Therefore, the
/// parameter vector of this model is empty.
///
template < class InputT>
class HierarchicalClustering : public AbstractClustering<InputT>
{
public:
	typedef AbstractClustering<InputT> base_type;
	typedef BinaryTree<InputT> tree_type;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// \brief Constructor
	///
	/// \par
	/// Construct a hierarchy of clusters from a binary tree.
	///
	/// \param  tree  tree object underlying the clustering
	HierarchicalClustering(const tree_type* tree)
	: mep_tree(tree){
		SHARK_CHECK(tree, "[HierarchicalClustering] Tree must not be NULL");
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "HierarchicalClustering"; }


	/// Return the number of clusters.
	std::size_t numberOfClusters() const{
		return (mep_tree->nodes() + 1) / 2;
	}

	/// Return the best matching cluster for very pattern in the batch.
	BatchOutputType hardMembership(BatchInputType const& patterns) const{
		std::size_t numPatterns = boost::size(patterns);
		BatchOutputType memberships(numPatterns);
		for(std::size_t i = 0; i != numPatterns; ++i){
			tree_type const* tree = mep_tree;
			memberships(i) = 0;
			while (tree->hasChildren()){
				if (tree->isLeft(get(patterns,i))){
					tree = tree->left();
				}
				else{
					memberships(i) += (tree->left()->nodes() + 1) / 2;
					tree = tree->right();
				}
			}
		}
		return memberships;
	}

	/// from IParameterizable
	RealVector parameterVector() const{
		return RealVector();
	}

	/// from IParameterizable
	void setParameterVector(RealVector const& newParameters){
		SHARK_ASSERT(newParameters.size() == 0);
	}

	/// from IParameterizable
	std::size_t numberOfParameters() const{
		return 0;
	}

protected:
	/// binary tree
	tree_type const* mep_tree;
};


}
#endif
