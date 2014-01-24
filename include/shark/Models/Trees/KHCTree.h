//===========================================================================
/*!
 * 
 * \file        KHCTree.h
 *
 * \brief       Tree for nearest neighbor search in kernel-induced feature spaces.
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_KHCTREE_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_KHCTREE_H


#include <shark/Models/Trees/BinaryTree.h>
#include <shark/LinAlg/Base.h>
#include <boost/array.hpp>

namespace shark {


///
/// \brief KHC-tree, a binary space-partitioning tree
///
/// \par
/// KHC-tree data structure for efficient nearest
/// neighbor searches in kernel-induced feature
/// spaces.
/// "KHC" stands for Kernel Hierarchical Clustering.
/// The space separation is based on distances to
/// cluster centers.
///
/// \par
/// The tree is constructed from data by splitting
/// along the pair of data points with largest
/// distance. This quantity is approximated using
/// a given number of randomly sampled pairs, which
/// is controlled by the template parameter
/// CuttingAccuracy.
///
/// \par
/// The bucket size for the tree is always one,
/// such that there is a unique point in each leaf
/// cell.
///
///Since the KHCTree needs direct access to the elements, it's template parameter is not the actual point type
///But the Range, the points are stored in. Be aware that this range should be a View when a Dataset is used as storage,
///since during construction, the KHC-Tree needs random access to the elements.
template <class Container, int CuttingAccuracy = 25>
class KHCTree : public BinaryTree<typename Container::value_type>
{
public:
	typedef IndexedIterator<typename boost::range_iterator<Container const>::type> const_iterator;
	typedef typename Container::value_type value_type;
	typedef AbstractKernelFunction<value_type> kernel_type;
	typedef BinaryTree<value_type> base_type;

	/// Construct the tree from data.
	/// It is assumed that the container exceeds
	/// the lifetime of the KHCTree (which holds
	/// only references to the points), and that
	/// the memory locations of the points remain
	/// unchanged.
	KHCTree(Container const& points, kernel_type const* kernel, TreeConstruction tc = TreeConstruction())
	: base_type(points.size())
	, mep_kernel(kernel)
	, m_normalInvNorm(1.0)
    {
		//create a list to the iterator elements as temporary storage
		//we need indexed operators to have a fast lookup of the position of the elements in the container
		std::vector<const_iterator> elements(m_size);
		boost::iota(elements,const_iterator(boost::begin(points),0));

		buildTree(tc,elements);

		//after the creation of the trees, the iterators in the array are sorted in the order,
		//they are referenced by the nodes.so we can create the indexList using the indizes of the iterators
		for(std::size_t i = 0; i != m_size; ++i){
			mp_indexList[i] = elements[i].index();
		}
	}


	/// \par
	/// Compute the squared distance of this cell to
	/// the given reference point, or alternatively
	/// a lower bound on this value.
	double squaredDistanceLowerBound(value_type const& reference) const{
		double dist = 0.0;
		KHCTree const* t = this;
		KHCTree const* p = (KHCTree const*)mep_parent;
		while (p != NULL){
			double v = p->distanceFromPlane(reference);
			if (t == p->mp_right)
				v = -v;
			if (v > dist)
				dist = v;
			t = p;
			p = (KHCTree const*)p->mep_parent;
		}
		return (dist * dist);
	}

protected:
	using base_type::mep_parent;
	using base_type::mp_left;
	using base_type::mp_right;
	using base_type::mp_indexList;
	using base_type::m_size;
	using base_type::m_nodes;

	/// (internal) construction of a non-root node
	KHCTree(KHCTree* parent, std::size_t* list, std::size_t size)
	: base_type(parent, list, size)
	, mep_kernel(parent->mep_kernel)
	, m_normalInvNorm(1.0)
	{ }

	/// (internal) construction method:
	/// median-cuts of the dimension with widest spread
	template<class Range>
	void buildTree(TreeConstruction tc, Range& points){
		typedef typename Range::iterator range_iterator;

		//check whether we are finished
		if (tc.maxDepth() == 0 || m_size <= tc.maxBucketSize()) {
			m_nodes = 1;
			return;
		}

		// use only a subset of size at most CuttingAccuracy
		// to estimate the vector along the longest
		// distance
		if (m_size <= CuttingAccuracy){
			calculateNormal(points);
		}
		else{
			boost::array<const_iterator,CuttingAccuracy> samples;
			for(std::size_t i = 0; i != CuttingAccuracy; i++)
				samples[i] = points[m_size * (2*i+1) / (2*CuttingAccuracy)];
			calculateNormal(samples);
		}

		//calculate the distance from the plane for every point in the list
		std::vector<double> distance(m_size);
		for(std::size_t i = 0; i != m_size; ++i){
			distance[i] = funct(*points[i]);
		}


		// split the list into sub-cells
		range_iterator split = this->splitList(distance,points);
		range_iterator begin = boost::begin(points);
		range_iterator end = boost::end(points);

		if (split == end) {//can't split points.
			m_nodes = 1;
			return;
		}

		// create sub-nodes
		std::size_t leftSize = split-begin;
		mp_left = new KHCTree(this, mp_indexList, leftSize);
		mp_right = new KHCTree(this, mp_indexList + leftSize, m_size - leftSize);

		// recurse
		boost::iterator_range<range_iterator> left(begin,split);
		boost::iterator_range<range_iterator> right(split,end);
		((KHCTree*)mp_left)->buildTree(tc.nextDepthLevel(),left);
		((KHCTree*)mp_right)->buildTree(tc.nextDepthLevel(),right);
		m_nodes = 1 + mp_left->nodes() + mp_right->nodes();
	}

	template<class Range>
	void calculateNormal(Range const& samples){
		std::size_t numSamples = shark::size(samples);
		double best_dist2 = -1.0;
		for (std::size_t i=1; i != numSamples; i++){
			for (std::size_t j = 0; j != i; j++){
				double dist2 = mep_kernel->featureDistanceSqr(*samples[i], *samples[j]);
				if (dist2 > best_dist2){
					best_dist2 = dist2;
					mep_positive = samples[i];
					mep_negative = samples[j];
				}
			}
		}
		m_normalInvNorm = 1.0 / std::sqrt(best_dist2);
		if (! (boost::math::isfinite)(m_normalInvNorm))
			m_normalInvNorm = 1.0;
	}



	/// function describing the separating hyperplane
	double funct(value_type const& reference) const{
		double result = mep_kernel->eval(*mep_positive, reference);
		result -= mep_kernel->eval(*mep_negative, reference);
		result *= m_normalInvNorm;
		return  result;
	}

	/// kernel function
	kernel_type const* mep_kernel;

	/// "positive" side cluster center
	const_iterator mep_positive;

	/// "negative" side cluster center
	const_iterator mep_negative;

	/// one divided by squared distance between "positive" and "negative" cluster center
	double m_normalInvNorm;
};


}
#endif
