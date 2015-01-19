//===========================================================================
/*!
 * 
 *
 * \brief       Efficient Nearest neighbor queries.
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_TREENEARESTNEIGHBORS_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_TREENEARESTNEIGHBORS_H


#include <boost/intrusive/rbtree.hpp>
#include <shark/Models/Trees/BinaryTree.h>
#include <shark/Algorithms/NearestNeighbors/AbstractNearestNeighbors.h>
#include <shark/Data/DataView.h>
namespace shark {


///
/// \brief Iterative nearest neighbors query.
///
/// \par
/// The IterativeNNQuery class (Iterative Nearest Neighbor
/// Query) allows the nearest neighbors of a reference point
/// to be queried iteratively. Given the reference point, a
/// query is set up that returns the nearest neighbor first,
/// then the second nearest neighbor, and so on.
/// Thus, nearest neighbor queries are treated in an "online"
/// fashion. The algorithm follows the paper (generalized to
/// arbitrary space-partitioning trees):
///
/// \par
/// Strategies for efficient incremental nearest neighbor search.
/// A. J. Broder. Pattern Recognition 23(1/2), pp 171-178, 1990.
///
/// \par
/// The algorithm is based on traversing a BinaryTree that
/// partitions the space into nested cells. The triangle
/// inequality is applied to exclude cells from the search.
/// Furthermore, candidate points are cached in a queue,
/// such that subsequent queries profit from points that
/// could not be excluded this way, but that did not turn
/// out the be the (current) nearest neighbor.
///
/// \par
/// The tree must have a bucket size of one, but leaf nodes
/// with multiple copies of the same point are allowed.
/// This means that the space partitioning must be carried
/// out to the finest possible scale.
///
/// The Data must be sotred in a random access container. This means that elements
/// have O(1) access time. This is crucial for the performance of the tree lookup.
/// When data is stored in a Data<T>, a View should be chosen as template parameter.
template <class DataContainer>
class IterativeNNQuery
{
public:
	typedef typename DataContainer::value_type value_type;
	typedef BinaryTree<value_type> tree_type;
	typedef AbstractKernelFunction<value_type> kernel_type;
	typedef std::pair<double, std::size_t> result_type;

	/// create a new query
	/// \param  tree    Underlying space-partitioning tree (this is assumed to persist for the lifetime of the query object).
	/// \param  data    Container holding the stored data which is referenced by the tree
	/// \param  point   Point whose nearest neighbors are to be found.
	IterativeNNQuery(tree_type const* tree, DataContainer const& data, value_type const& point)
	: m_data(data)
	, m_reference(point)
	, m_nextIndex(0)
	, mp_trace(NULL)
	, mep_head(NULL)
	, m_squaredRadius(0.0)
	, m_neighbors(0)
	{
		// Initialize the recursion trace: descend to the
		// leaf covering the reference point and queue it.
		// The parent of this leaf becomes the "head".
		mp_trace = new TraceNode(tree, NULL, m_reference);
		TraceNode* tn = mp_trace;
		while (tree->hasChildren())
		{
			tn->createLeftNode(tree, m_data, m_reference);
			tn->createRightNode(tree, m_data, m_reference);
			bool left = tree->isLeft(m_reference);
			tn = (left ? tn->mep_left : tn->mep_right);
			tree = (left ? tree->left() : tree->right());
		}
		mep_head = tn->mep_parent;
		insertIntoQueue((TraceLeaf*)tn);
		m_squaredRadius = mp_trace->squaredRadius(m_reference);
	}

	/// destroy the query object and its internal data structures
	~IterativeNNQuery() {
		m_queue.clear();
		delete mp_trace;
	}


	/// return the number of neighbors already found
	std::size_t neighbors() const {
		return m_neighbors;
	}

	/// find and return the next nearest neighbor
	result_type next() {
		if (m_neighbors >= mp_trace->m_tree->size()) 
			throw SHARKEXCEPTION("[IterativeNNQuery::next] no more neighbors available");

		assert(! m_queue.empty());

		// Check whether the current node has points
		// left, or whether it should be discarded.
		if (m_neighbors > 0){
			TraceLeaf& q = *m_queue.begin();
			if (m_nextIndex < q.m_tree->size()){
				return getNextPoint(q);
			}
			else
				m_queue.erase(q);
		}
		if (m_queue.empty() || (*m_queue.begin()).m_squaredPtDistance > m_squaredRadius){
			// enqueue more points
			TraceNode* tn = mep_head;
			while (tn != NULL){
				enqueue(tn);
				if (tn->m_status == COMPLETE) mep_head = tn->mep_parent;
				tn = tn->mep_parent;
			}

			// re-compute the radius
			m_squaredRadius = mp_trace->squaredRadius(m_reference);
		}
		m_nextIndex = 0;
		++m_neighbors;
		return getNextPoint(*m_queue.begin());
	}

	/// return the size of the queue,
	/// which is a measure of the
	/// overhead of the search
	std::size_t queuesize() const{ 
		return m_queue.size();
	}

private:

	/// status of a TraceNode object during the search
	enum Status
	{
		NONE,            // no points of this node have been queued yet
		PARTIAL,         // some of the points of this node have been queued
		COMPLETE,        // all points of this node have been queued
	};

	/// The TraceNode class builds up a tree during
	/// the search. This tree covers only those parts
	/// of the space partirioning tree that need to be
	/// traversed in order to find the next nearest
	/// neighbor.
	class TraceNode
	{
	public:
		/// Constructor
		TraceNode(tree_type const* tree, TraceNode* parent, value_type const& reference)
		: m_tree(tree)
		, m_status(NONE)
		, mep_parent(parent)
		, mep_left(NULL)
		, mep_right(NULL)
		, m_squaredDistance(tree->squaredDistanceLowerBound(reference))
		{ }

		/// Destructor
		virtual ~TraceNode()
		{
			if (mep_left != NULL) delete mep_left;
			if (mep_right != NULL) delete mep_right;
		}
		
		void createLeftNode(tree_type const* tree, DataContainer const& data, value_type const& reference){
			if (tree->left()->hasChildren())
				mep_left = new TraceNode(tree->left(), this, reference);
			else
				mep_left = new TraceLeaf(tree->left(), this, data, reference);
		}
		void createRightNode(tree_type const* tree, DataContainer const& data, value_type const& reference){
			if (tree->right()->hasChildren())
				mep_right = new TraceNode(tree->right(), this, reference);
			else
				mep_right = new TraceLeaf(tree->right(), this, data, reference);
		}

		/// Compute the squared distance of the area not
		/// yet covered by the queue to the reference point.
		/// This is also referred to as the squared "radius"
		/// of the area covered by the queue (in fact, it is
		/// the radius of the largest sphere around the
		/// reference point that fits into the covered area).
		double squaredRadius(value_type const& ref) const{
			if (m_status == NONE) return m_squaredDistance;
			else if (m_status == PARTIAL)
			{
				double l = mep_left->squaredRadius(ref);
				double r = mep_right->squaredRadius(ref);
				return std::min(l, r);
			}
			else return 1e100;
		}

		/// node of the tree
		tree_type const* m_tree;

		/// status of the search
		Status m_status;

		/// parent node
		TraceNode* mep_parent;

		/// "left" child
		TraceNode* mep_left;

		/// "right" child
		TraceNode* mep_right;

		/// squared distance of the box to the reference point
		double m_squaredDistance;
	};

	/// hook type for intrusive container
	typedef boost::intrusive::set_base_hook<> HookType;

	/// Leaves of the three have three roles:
	/// (1) they are tree nodes holding exactly one point
	///     (possibly multiple copies of this point),
	/// (2) they know the distance of their point to the
	///     reference point,
	/// (3) they can be added to the candidates queue.
	class TraceLeaf : public TraceNode, public HookType
	{
	public:
		/// Constructor
		TraceLeaf(tree_type const* tree, TraceNode* parent, DataContainer const& data, value_type const& ref)
		: TraceNode(tree, parent, ref){
			//check whether the tree uses a differen metric than a linear one.
			if(tree->kernel() != NULL)
				m_squaredPtDistance = tree->kernel()->featureDistanceSqr(data[tree->index(0)], ref);
			else
				m_squaredPtDistance = distanceSqr(data[tree->index(0)], ref);
		}

		/// Destructor
		~TraceLeaf() { }


		/// Comparison by distance, ties are broken arbitrarily,
		/// but deterministically, by tree node pointer.
		inline bool operator < (TraceLeaf const& rhs) const{
			if (m_squaredPtDistance == rhs.m_squaredPtDistance) 
				return (this->m_tree < rhs.m_tree);
			else
				return (m_squaredPtDistance < rhs.m_squaredPtDistance);
		}

		/// Squared distance of the single point in the leaf to the reference point.
		double m_squaredPtDistance;
	};

	/// insert a point into the queue
	void insertIntoQueue(TraceLeaf* leaf){
		m_queue.insert_unique(*leaf);

		// traverse up the tree, updating the state
		TraceNode* tn = leaf;
		tn->m_status = COMPLETE;
		while (true){
			TraceNode* par = tn->mep_parent;
			if (par == NULL) break;
			if (par->m_status == NONE){
				par->m_status = PARTIAL;
				break;
			}
			else if (par->m_status == PARTIAL){
				if (par->mep_left == tn){
					if (par->mep_right->m_status == COMPLETE) par->m_status = COMPLETE;
					else break;
				}
				else{
					if (par->mep_left->m_status == COMPLETE) par->m_status = COMPLETE;
					else break;
				}
			}
			tn = par;
		}
	}

	result_type getNextPoint(TraceLeaf const& leaf){
		double dist = std::sqrt(leaf.m_squaredPtDistance);
		std::size_t index = leaf.m_tree->index(m_nextIndex);
		++m_nextIndex;
		return std::make_pair(dist,index);
	}

	/// Recursively descend the node and enqueue
	/// all points in cells intersecting the
	/// current bounding sphere.
	void enqueue(TraceNode* tn){
		// check whether this node needs to be enqueued
		if (tn->m_status == COMPLETE) return;
		if (! m_queue.empty() && tn->m_squaredDistance >= (*m_queue.begin()).m_squaredPtDistance) return;

		const tree_type* tree = tn->m_tree;
		if (tree->hasChildren()){
			// extend the tree at need
			if (tn->mep_left == NULL){
				tn->createLeftNode(tree,m_data,m_reference);
			}
			if (tn->mep_right == NULL){
				tn->createRightNode(tree,m_data,m_reference);
			}

			// first descend into the closer sub-tree
			if (tree->isLeft(m_reference))
			{
				// left first
				enqueue(tn->mep_left);
				enqueue(tn->mep_right);
			}
			else
			{
				// right first
				enqueue(tn->mep_right);
				enqueue(tn->mep_left);
			}
		}
		else
		{
			TraceLeaf* leaf = (TraceLeaf*)tn;
			insertIntoQueue(leaf);
		}
	}

	/// the queue is a self-balancing tree of sorted entries
	typedef boost::intrusive::rbtree<TraceLeaf> QueueType;


	///\brief Datastorage to lookup the points referenced by the space partitioning tree.
	DataContainer const& m_data;

	/// reference point for this query
	value_type m_reference;

	/// queue of candidates
	QueueType m_queue;

	/// index of the next not yet returned element
	/// of the current leaf.
	std::size_t m_nextIndex;

	/// recursion trace tree
	TraceNode* mp_trace;

	/// "head" of the trace tree. This is the
	/// node containing the reference point,
	/// but so high up in the tree that it is
	/// not fully queued yet.
	TraceNode* mep_head;

	/// squared radius of the "covered" area
	double m_squaredRadius;

	/// number of neighbors already returned
	std::size_t m_neighbors;
};


///\brief Nearest Neighbors implementation using binary trees
///
/// Returns the labels and distances of the k nearest neighbors of a point.
template<class InputType, class LabelType>
class TreeNearestNeighbors:public AbstractNearestNeighbors<InputType,LabelType>
{
private:
	typedef AbstractNearestNeighbors<InputType,LabelType> base_type;

public:
	typedef LabeledData<InputType, LabelType> Dataset;
	typedef BinaryTree<InputType> Tree;
	typedef typename base_type::DistancePair DistancePair;
	typedef typename Batch<InputType>::type BatchInputType;

	TreeNearestNeighbors(Dataset const& dataset, Tree const* tree)
	: m_dataset(dataset), m_inputs(dataset.inputs()), m_labels(dataset.labels()),mep_tree(tree)
	{ }

	///\brief returns the k nearest neighbors of the point
	std::vector<DistancePair> getNeighbors(BatchInputType const& patterns, std::size_t k)const{
		std::size_t numPoints = shark::size(patterns);
		std::vector<DistancePair> results(k*numPoints);
		for(std::size_t p = 0; p != numPoints; ++p){
			IterativeNNQuery<DataView<Data<InputType> const> > query(mep_tree, m_inputs, get(patterns, p));
			//find the neighbors using the queries
			for(std::size_t i = 0; i != k; ++i){
				typename IterativeNNQuery<DataView<Data<InputType> const> >::result_type result = query.next();
				results[i+p*k].key=result.first;
				results[i+p*k].value= m_labels[result.second]; 
			}
		}
		return results;
	}

	LabeledData<InputType,LabelType>const& dataset()const {
		return m_dataset;
	}

private:
	Dataset const& m_dataset;
	DataView<Data<InputType> const> m_inputs;
	DataView<Data<LabelType> const> m_labels;
	Tree const* mep_tree;
	
};


}
#endif
