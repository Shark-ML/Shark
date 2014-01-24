//===========================================================================
/*!
 * 
 * \file        KDTree.h
 *
 * \brief       Tree for nearest neighbor search in low dimensions.
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_KDTREE_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_KDTREE_H


#include <shark/Models/Trees/BinaryTree.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/Math.h>
namespace shark {


///
/// \brief KD-tree, a binary space-partitioning tree
///
/// \par
/// KD-tree data structure for efficient nearest
/// neighbor searches in low-dimensional spaces.
/// Low-dimensional means << 10 dimensions.
///
/// \par
/// The tree is constructed from data by splitting
/// the dimension with largest extent (in the data
/// covered, not space represented by the cell),
/// recursively. An approximate median is used as
/// the cutting point, where the maximal number of
/// points used to estimate the median can be
/// controlled with the template parameter
/// MedianAccuracy.
///
/// \par
/// The bucket size for the tree is aleays one,
/// such that there is a unique point in each leaf
/// cell.
///
template <class InputT>
class KDTree : public BinaryTree<InputT>
{
	typedef KDTree<InputT> self_type;
	typedef BinaryTree<InputT> base_type;
public:

	/// Construct the tree from data.
	/// It is assumed that the container exceeds
	/// the lifetime of the KDTree (which holds
	/// only references to the points), and that
	/// the memory locations of the points remain
	/// unchanged.
	KDTree(Data<InputT> const& dataset, TreeConstruction tc = TreeConstruction())
	: base_type(dataset.numberOfElements())
	, m_cutDim(0xffffffff){
		typedef DataView<Data<RealVector> const> PointSet;
		PointSet points(dataset);
		//create a list to the iterator elements as temporary storage
		std::vector<typename boost::range_iterator<PointSet>::type> elements(m_size);
		boost::iota(elements,boost::begin(points));

		buildTree(tc,elements);

		//after the creation of the trees, the iterators in the array are sorted in the order, 
		//they are referenced by the nodes.so we can create the indexList using the indizes of the iterators
		for(std::size_t i = 0; i != m_size; ++i){
			mp_indexList[i] = elements[i].index();
		}
	}


	/// lower bound on the box-shaped
	/// space represented by this cell
	double lower(std::size_t dim) const{
		self_type* parent = static_cast<self_type*>(mep_parent);
		if (parent == NULL) return -1e100;

		if (parent->m_cutDim == dim && parent->mp_right == this)
			return parent->threshold();
		else
			return parent->lower(dim);
	}

	/// upper bound on the box-shaped
	/// space represented by this cell
	double upper(std::size_t dim) const{
		self_type* parent = static_cast<self_type*>(mep_parent);
		if (parent == NULL) return +1e100;

		if (parent->m_cutDim == dim && parent->mp_left == this) 
			return parent->threshold();
		else 
			return parent->upper(dim);
	}

	/// \par
	/// Compute the squared Euclidean distance of
	/// this cell to the given reference point, or
	/// alternatively a lower bound on this value.
	///
	/// \par
	/// In the case of the kd-tree the computation
	/// is exact, however, only a lower bound is
	/// required in general, and other space
	/// partitioning trees used in the future may
	/// only be able to provide a lower bound, at
	/// least with reasonable computational efforts.
	double squaredDistanceLowerBound(InputT const& reference) const
	{
		double ret = 0.0;
		for (std::size_t d = 0; d != reference.size(); d++)
		{
			double v = reference(d);
			double l = lower(d);
			double u = upper(d);
			if (v < l){
				ret += sqr(l-v);
			}
			else if (v > u){
				ret += sqr(v-u);
			}
		}
		return ret;
	}

#if 0
	// debug code, please ignore
	void print(unsigned int ident = 0) const
	{
		if (this->isLeaf())
		{
			for (unsigned int j=0; j<m_size; j++)
			{
				for (unsigned int i=0; i<ident; i++) printf("  ");
				printf("index: %d\n", (int)this->index(j));
			}
		}
		else
		{
			for (unsigned int i=0; i<ident; i++) printf("  ");
			printf("x[%d] < %g\n", (int)m_cutDim, this->threshold());
			for (unsigned int i=0; i<ident; i++) printf("  ");
			printf("[%d]\n", (int)mp_left->size());
			((self_type*)mp_left)->print(ident + 1);
			for (unsigned int i=0; i<ident; i++) printf("  ");
			printf("[%d]\n", (int)mp_right->size());
			((self_type*)mp_right)->print(ident + 1);
		}
	}
#endif

protected:
	using base_type::mep_parent;
	using base_type::mp_left;
	using base_type::mp_right;
	using base_type::mp_indexList;
	using base_type::m_size;
	using base_type::m_nodes;

	/// (internal) construction of a non-root node
	KDTree(KDTree* parent, std::size_t* list, std::size_t size)
	: base_type(parent, list, size)
	, m_cutDim(0xffffffff)
	{ }

	/// (internal) construction method:
	/// median-cuts of the dimension with widest spread
	template<class Range>
	void buildTree(TreeConstruction tc, Range& points){
		typedef typename boost::range_iterator<Range>::type iterator;

		iterator begin = boost::begin(points);
		iterator end = boost::end(points);

		if (tc.maxDepth() == 0 || m_size <= tc.maxBucketSize()){
			m_nodes = 1; 
			return; 
		}

		m_cutDim = calculateCuttingDimension(points);

		// calculate the distance of the boundary for every point in the list
		std::vector<double> distance(m_size);
		iterator point = begin;
		for(std::size_t i = 0; i != m_size; ++i,++point){
			distance[i] = get(**point,m_cutDim);
		}

		// split the list into sub-cells
		iterator split = this->splitList(distance,points);
		if (split == end){
			m_nodes = 1;
			return; 
		}
		std::size_t leftSize = split-begin;

		// create sub-nodes
		mp_left = new KDTree(this, mp_indexList, leftSize);
		mp_right = new KDTree(this, mp_indexList + leftSize, m_size - leftSize);

		// recurse
		boost::iterator_range<iterator> left(begin,split);
		boost::iterator_range<iterator> right(split,end);
		((KDTree*)mp_left)->buildTree(tc.nextDepthLevel(), left);
		((KDTree*)mp_right)->buildTree(tc.nextDepthLevel(), right);
		m_nodes = 1 + mp_left->nodes() + mp_right->nodes();
	}

	///\brief Calculate the dimension which should be cut by this node
	///
	///The KD-Tree calculates the Axis-Aligned-Bounding-Box surrounding the points
	///and splits the longest dimension
	template<class Range>
	std::size_t calculateCuttingDimension(Range const& points)const{
		typedef typename boost::range_iterator<Range const>::type iterator;

		iterator begin = boost::begin(points);
		iterator end = boost::end(points);

		// calculate bounding box of the data
		InputT L = **begin;
		InputT U = **begin;
		std::size_t dim = L.size();
		iterator point = begin;
		++point;
		for (std::size_t i=1; i != m_size; ++i,++point){
			for (std::size_t d = 0; d != dim; d++){
				double v = (**point)[d];
				if (v < L[d]) L[d] = v;
				if (v > U[d]) U[d] = v;
			}
		}

		// find the longest edge of the bounding box
		std::size_t cutDim = 0;
		double extent = U[0] - L[0];
		for (std::size_t d = 1; d != dim; d++)
		{
			double e = U[d] - L[d];
			if (e > extent)
			{
				extent = e;
				cutDim = d;
			}
		}
		return cutDim;
	}

	/// Function describing the separating hyperplane
	/// as its zero set. It is guaranteed that the
	/// gradient has norm one, thus the absolute value
	/// of the function indicates distance from the
	/// hyperplane.
	double funct(InputT const& reference) const{
		return reference[m_cutDim];
	}

	/// split/cut dimension of this node
	std::size_t m_cutDim;
};


}
#endif
