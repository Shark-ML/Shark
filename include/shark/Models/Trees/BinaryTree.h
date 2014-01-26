//===========================================================================
/*!
 * 
 *
 * \brief       Binary space-partitioning tree of data points.
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_BINARYTREE_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_BINARYTREE_H


#include <shark/Core/Exception.h>
#include <shark/Core/utility/functional.h>
#include <shark/Core/utility/KeyValuePair.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>

#include <boost/range/algorithm_ext/iota.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
namespace shark {


/// \brief Stopping criteria for tree construction.
///
/// \par
/// Conditions for automatic tree construction.
/// The data structure allows to specify a maximum
/// bucket size (number of instances represented
/// by a leaf), and a maximum tree depth.
///
/// \par
/// Note: If a data instance appears more often in
/// a dataset than specified by the maximum bucket
/// size then this condition will be violated; this
/// is because a space partitioning tree has no
/// means of separating a single point.
///
class TreeConstruction
{
public:
	/// \brief Default constructor: only stop at trivial leaves
	TreeConstruction()
	: m_maxDepth(0xffffffff)
	, m_maxBucketSize(1)
	{ }

	/// \brief Copy constructor.
	TreeConstruction(TreeConstruction const& other)
	: m_maxDepth(other.m_maxDepth)
	, m_maxBucketSize(other.m_maxBucketSize)
	{ }

	/// \brief Constructor.
	///
	/// \param  maxDepth       stop as soon as the given tree depth is reached (zero means unrestricted)
	/// \param  maxBucketSize  stop as soon as a node holds at most the bucket size of data points (zero means unrestricted)
	TreeConstruction(unsigned int maxDepth, unsigned int maxBucketSize)
	: m_maxDepth(maxDepth ? maxDepth : 0xffffffff)
	, m_maxBucketSize(maxBucketSize ? maxBucketSize : 1)
	{ }


	/// return a TreeConstruction object with maxDepth reduced by one
	TreeConstruction nextDepthLevel() const
	{ return TreeConstruction(m_maxDepth - 1, m_maxBucketSize); }


	/// return maximum depth of the tree
	unsigned int maxDepth() const
	{ return m_maxDepth; }

	/// return maximum "size" of a leaf node
	unsigned int maxBucketSize() const
	{ return m_maxBucketSize; }

protected:
	/// maximum depth of the tree
	unsigned int m_maxDepth;

	/// maximum "size" of a leaf node
	unsigned int m_maxBucketSize;
};


///
/// \brief Super class of binary space-partitioning trees.
///
/// \par
/// This class represents a generic node in a binary
/// space-partitioning tree. At each junction the
/// space cell represented by the parent node is
/// split into sub-cells by thresholding a real-valued
/// function. Different sub-classes implement different
/// such functions. The absolute value of the function
/// minus the threshold m_threshold must represent the
/// distance to the separating hyper-surface in the
/// underlying metric. This allows for linear separation,
/// but also for kernel-induced feature spaces and other
/// embeddings.
template <class InputT>
class BinaryTree
{
public:
	typedef InputT value_type;

	/// \brief Root node constructor: build the tree from data.
	///
	/// Please refer the specific sub-classes such as KDTree
	/// for examples of how the binary tree is built.
	///
	BinaryTree(std::size_t size)
	: mep_parent(NULL)
	, mp_left(NULL)
	, mp_right(NULL)
	, mp_indexList(NULL)
	, m_size(size)
	, m_nodes(0)
	, m_threshold(0.0)
	{
		SHARK_ASSERT(m_size > 0);

		// prepare list of index/pointer pairs to be shared among the whole tree
		mp_indexList = new std::size_t[m_size];
		boost::iota(boost::make_iterator_range(mp_indexList,mp_indexList+m_size),0);
	}

	/// Destroy the tree and its internal data structures
	virtual ~BinaryTree()
	{
		if (mp_left != NULL) delete mp_left;
		if (mp_right != NULL) delete mp_right;
		if (mep_parent == NULL) delete [] mp_indexList;
	}


	// binary tree structure

	/// parent node
	BinaryTree* parent()
	{ return mep_parent; }
	/// parent node
	const BinaryTree* parent() const
	{ return mep_parent; }

	/// Does this node have children?
	/// Opposite of isLeaf()
	bool hasChildren() const
	{ return (mp_left != NULL); }

	/// Is this a leaf node?
	/// Opposite of hasChildren()
	bool isLeaf() const
	{ return (mp_left == NULL); }

	/// "left" sub-node of the binary tree
	BinaryTree* left()
	{ return mp_left; }
	/// "left" sub-node of the binary tree
	const BinaryTree* left() const
	{ return mp_left; }

	/// "right" sub-node of the binary tree
	BinaryTree* right()
	{ return mp_right; }
	/// "right" sub-node of the binary tree
	const BinaryTree* right() const
	{ return mp_right; }

	/// number of points inside the space represented by this node
	std::size_t size() const
	{ return m_size; }

	/// number of sub-nodes in this tree (including the node itself)
	std::size_t nodes() const
	{ return m_nodes; }

	std::size_t index(std::size_t point)const{
		return mp_indexList[point];
	}


	// partition represented by this node

	/// \brief Function describing the separation of space.
	///
	/// \par
	/// This function is shifted by subtracting the
	/// threshold from the virtual function "funct" (which
	/// acts as a "decision" function to split space into
	/// sub-cells).
	/// The result of this function describes "signed"
	/// distance to the separation boundary, and the two
	/// cells are thresholded at zero. We obtain the two
	/// cells:<br/>
	/// left ("negative") cell: {x | distance(x) < 0}<br/>
	/// right ("positive") cell {x | distance(x) >= 0}
	double distanceFromPlane(value_type const& point) const{
		return funct(point) - m_threshold;
	}

	/// \brief Separation threshold.
	double threshold() const{
		return m_threshold;
	}

	/// return true if the reference point belongs to the
	/// "left" sub-node, or to the "negative" sub-cell.
	bool isLeft(value_type const& point) const
	{ return (funct(point) < m_threshold); }

	/// return true if the reference point belongs to the
	/// "right" sub-node, or to the "positive" sub-cell.
	bool isRight(value_type const& point) const
	{ return (funct(point) >= m_threshold); }

	/// \brief If the tree uses a kernel metric, returns a pointer to the kernel object, else NULL.
	virtual AbstractKernelFunction<value_type> const* kernel()const{
		//default is no kernel metric
		return NULL;
	}


	/// \brief Compute lower bound on the squared distance to the space cell.
	///
	/// \par
	/// For efficient use of the triangle inequality
	/// the space cell represented by each node needs
	/// to provide (a lower bound on) the distance of
	/// a query point to the space cell represented by
	/// this node. Search is fast if this bound is
	/// tight.
	virtual double squaredDistanceLowerBound(value_type const& point) const = 0;

#if 0
	// debug code, please ignore
	void print(unsigned int ident = 0) const
	{
		if (isLeaf())
		{
			for (unsigned int j=0; j<size(); j++)
			{
				for (unsigned int i=0; i<ident; i++) printf("  ");
				printf("index: %d\n", (int)index(j));
			}
		}
		else
		{
			for (unsigned int i=0; i<ident; i++) printf("  ");
			printf("[%d]\n", (int)mp_left->size());
			mp_left->print(ident + 1);
			for (unsigned int i=0; i<ident; i++) printf("  ");
			printf("[%d]\n", (int)mp_right->size());
			mp_right->print(ident + 1);
		}
	}
#endif

protected:
	/// \brief Sub-node constructor
	///
	/// \par
	/// Initialize a sub-node
	BinaryTree(BinaryTree* parent, std::size_t* list, std::size_t size)
	: mep_parent(parent)
	, mp_left(NULL)
	, mp_right(NULL)
	, mp_indexList(list)
	, m_size(size)
	, m_nodes(0)
	{}


	/// \brief Function describing the separation of space.
	///
	/// \par
	/// This is the primary interface for customizing
	/// sub-classes.
	///
	/// \par
	/// This function splits the space represented by the
	/// node by thresholding at m_threshold. The "negative"
	/// cell, represented in the "left" sub-node, represents
	/// the space {x | funct(x) < threshold}. The
	/// "positive" cell, represented by the "right"
	/// sub-node, represents {x | funct(x) >= threshold}.
	/// The function needs to be normalized such that
	/// |f(x) - f(y)| is the distance between
	/// {z | f(z) = f(x)} and {z | f(z) = f(y)}, w.r.t.
	/// the distance function also used by the virtual
	/// function squaredDistanceLowerBound. The exact
	/// distance function does not matter as long as
	/// the same distance function is used in both cases.
	virtual double funct(value_type const& point) const = 0;

	/// \brief Split the data in the point list and calculate the treshold accordingly
	///
	/// The method computes the optimal threshold given the distance of every element of
	/// the set and partitions the point range accordingly
	/// @param values the value of every point returned by funct(point)
	/// @param points the points themselves
	/// @returns returns the position were the point list was split
	template<class Range1, class Range2>
	typename boost::range_iterator<Range2>::type splitList (Range1& values, Range2& points){
		typedef typename boost::range_iterator<Range1>::type iterator1;
		typedef typename boost::range_iterator<Range2>::type iterator2;

		iterator1 valuesBegin = boost::begin(values);
		iterator1 valuesEnd = boost::end(values);

		//KeyValueRange<iterator1,iterator2> kvrange = ;
		std::pair<iterator1, iterator2> splitpoint = partitionEqually(zipKeyValuePairs(values,points)).iterators();
		iterator1 valuesSplitpoint = splitpoint.first;
		iterator2 pointsSplitpoint = splitpoint.second;
		if (valuesSplitpoint == valuesEnd) {
			// partitioning failed, all values are equal :(
			m_threshold = *valuesBegin;
			return splitpoint.second;
		}

		// We don't want the threshold to be the value of an element but always in between two of them.
		// This ensures that no point of the training set lies on the boundary. This leeds to more stable
		// results. So we use the mean of the found splitpoint and the nearest point on the other side
		// of the boundary.
		double maximum = *std::max_element(valuesBegin, valuesSplitpoint);
		m_threshold = 0.5*(maximum + *valuesSplitpoint);

		return pointsSplitpoint;
	}

	/// parent node
	BinaryTree* mep_parent;

	/// "left" child node
	BinaryTree* mp_left;

	/// "right" child node
	BinaryTree* mp_right;

	/// list of indices to points in the dataset
	std::size_t* mp_indexList;

	/// number of points in this node
	std::size_t m_size;

	/// number of nodes in the sub-tree represented by this node
	std::size_t m_nodes;

	/// threshold for the separating function
	double m_threshold;

};


}
#endif
