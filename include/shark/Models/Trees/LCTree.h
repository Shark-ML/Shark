//===========================================================================
/*!
 * 
 * \file        LCTree.h
 *
 * \brief       Tree for nearest neighbor search in data with low embedding dimension.
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_LCTREE_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_LCTREE_H


#include <shark/Models/Trees/BinaryTree.h>
#include <shark/Data/DataView.h>
#include <shark/LinAlg/Base.h>
#include <boost/array.hpp>

namespace shark {


///
/// \brief LC-tree, a binary space-partitioning tree
///
/// \par
/// LC-tree data structure for efficient nearest
/// neighbor searches in possibly high-dimensional
/// spaces, but with low-dimensional effective data
/// dimension (means << 10 dimensions).
/// "LC" stands for Linear Cut.
///
/// This tree requires the existence of a function
/// inner_prod computing the standard inner product
/// of two objects of type VectorType, and a function
/// distanceSqr computing the squared Euclidean
/// distance between two vectors.
///
/// \par
/// The tree is constructed from data by splitting
/// the direction with largest extent (in the data
/// covered, not space represented by the cell),
/// recursively. Approximate direction and median
/// are used to determine the cutting hyperplane,
/// where the maximal number of points used to
/// estimate the median can be controlled with the
/// template parameter CuttingAccuracy.
///
/// \par
/// The bucket size for the tree is always one,
/// such that there is a unique point in each leaf
/// cell.
///
template <class VectorType = RealVector, int CuttingAccuracy = 25>
class LCTree : public BinaryTree<VectorType>
{
	typedef BinaryTree<VectorType> base_type;
public:
	/// Construct the tree from data.
	/// It is assumed that the container exceeds
	/// the lifetime of the LCTree (which holds
	/// only references to the points), and that
	/// the memory locations of the points remain
	/// unchanged.
	LCTree(Data<RealVector> const& dataset, TreeConstruction tc = TreeConstruction())
	: base_type(dataset.numberOfElements())
	, m_normal(dataDimension(dataset)){
		typedef DataView<Data<RealVector> const> PointSet;
		PointSet points(dataset);
		//create a list to the iterator elements as temporary storage
		//we need indexed operators to have a fast lookup of the position of the elements in the container
		typedef IndexedIterator<typename boost::range_iterator<PointSet>::type> iterator;
		std::vector<iterator> elements(m_size);
		boost::iota(elements,iterator(boost::begin(points),0));

		buildTree(tc,elements);
		//after the creation of the trees, the iterators in the array are sorted in the order, 
		//they are referenced by the nodes.so we can create the indexList using the indizes of the iterators
		for(std::size_t i = 0; i != m_size; ++i){
			mp_indexList[i] = elements[i].index();
		}
	}


	/// \par
	/// Compute the squared Euclidean distance of
	/// this cell to the given reference point, or
	/// alternatively a lower bound on this value.
	double squaredDistanceLowerBound(VectorType const& reference) const{
		double dist = 0.0;
		LCTree const* t = this;
		LCTree const* p = (LCTree const*)mep_parent;
		while (p != NULL)
		{
			double v = p->distanceFromPlane(reference);
			if (t == p->mp_right) 
				v = -v;
			if (v > dist) 
				dist = v;
			t = p;
			p = (LCTree const*)p->mep_parent;
		}
		return dist * dist;
	}

protected:
	using base_type::mep_parent;
	using base_type::mp_left;
	using base_type::mp_right;
	using base_type::mp_indexList;
	using base_type::m_size;
	using base_type::m_nodes;

	/// (internal) construction of a non-root node
	LCTree(LCTree* parent, std::size_t* list, std::size_t size)
	: base_type(parent, list, size){}

	/// (internal) construction method:
	/// median-cuts of the dimension with widest spread
	template<class Range>
	void buildTree(TreeConstruction tc, Range& points){
		typedef typename Range::value_type pointIterator;
		typedef typename Range::iterator iterator;
		
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
			boost::array<pointIterator,CuttingAccuracy> samples;
			for(std::size_t i = 0; i != CuttingAccuracy; i++) 
				samples[i] = points[m_size * (2*i+1) / (2*CuttingAccuracy)];
			calculateNormal(samples);
		}

		//calculate the distance from the plane for every point in the list
		std::vector<double> distance(m_size);
		for(std::size_t i = 0; i != m_size; ++i){
			distance[i] = inner_prod(m_normal, *points[i]);
		}
		
		
		// split the list into sub-cells
		iterator split = this->splitList(distance,points);
		iterator begin = boost::begin(points);
		iterator end = boost::end(points);
		
		if (split == end) {//can't split points. 
			m_nodes = 1;
			return; 
		}

		// create sub-nodes
		std::size_t leftSize = split-begin;
		mp_left = new LCTree(this, mp_indexList, leftSize);
		mp_right = new LCTree(this, mp_indexList + leftSize, m_size - leftSize);

		// recurse
		boost::iterator_range<iterator> left(begin,split);
		boost::iterator_range<iterator> right(split,end);
		((LCTree*)mp_left)->buildTree(tc.nextDepthLevel(),left);
		((LCTree*)mp_right)->buildTree(tc.nextDepthLevel(),right);
		m_nodes = 1 + mp_left->nodes() + mp_right->nodes();
	}

	/// function describing the separating hyperplane
	double funct(VectorType const& reference) const{ 
		return inner_prod(m_normal, reference);
	}
	
	//find the longest distance between vectors in the sample set and calculate
	//the normal along this direction
	template<class Range>
	void calculateNormal(Range const& samples){
		std::size_t numSamples = shark::size(samples);
		std::size_t besti = 0;
		std::size_t bestj = 0;
		double best_dist2 = -1.0;
		for (std::size_t i = 1; i != numSamples; i++){
			for (std::size_t j = 0; j != i; j++){
				double dist2 = distanceSqr(*samples[i], *samples[j]);
				if (dist2 > best_dist2){
					best_dist2 = dist2;
					besti = i;
					bestj = j;
				}
			}
		}
		double factor = 1.0 / std::sqrt(best_dist2);
		if (! (boost::math::isfinite)(factor))
			factor = 1.0;
				 
		m_normal = factor * (*samples[besti] -  *samples[bestj]);
	}

	/// split/cut normal vector of this node
	VectorType m_normal;
};


}
#endif
