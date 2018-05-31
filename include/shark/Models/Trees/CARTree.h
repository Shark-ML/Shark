//===========================================================================
/*!
 * 
 *
 * \brief       Cart Classifier
 * 
 * 
 *
 * \author      K. N. Hansen, J. Kremer
 * \date        2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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

#ifndef SHARK_MODELS_TREES_CARTree_H
#define SHARK_MODELS_TREES_CARTree_H


#include <shark/Models/AbstractModel.h>                             
#include <shark/Data/Dataset.h>
namespace shark {


///
/// \brief Classification and Regression Tree.
///
/// \par
/// The CARTree predicts a class label using a decision tree
/// \ingroup models
template<class LabelType>
class CARTree : public AbstractModel<RealVector,LabelType>
{
private:
	typedef AbstractModel<RealVector, LabelType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	
	struct Node{
		std::size_t attributeIndex;
		double attributeValue;
		std::size_t leftId;
		std::size_t rightIdOrIndex;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version){
			ar & attributeIndex;
			ar & attributeValue;
			ar & leftId;
			ar & rightIdOrIndex;///< either id of right node or index to label array
		}
	};
	typedef std::vector<Node> TreeType;
	

	/// Constructor
	CARTree(): m_inputDimension(0){}
	
	CARTree(std::size_t inputDimension, Shape const& outputShape)
	: m_inputDimension(inputDimension)
	, m_outputShape(outputShape){}


	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CARTree"; }

	boost::shared_ptr<State> createState() const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;
	/// \brief Evaluate the Tree on a batch of patterns
	void eval(BatchInputType const& patterns, BatchOutputType & outputs) const{
		std::size_t numPatterns = patterns.size1();
		//evaluate the first pattern alone and create the batch output from that
		LabelType const& firstResult = evalPattern(row(patterns,0));
		outputs = Batch<LabelType>::createBatch(firstResult,numPatterns);
		getBatchElement(outputs,0) = firstResult;
		
		//evaluate the rest
		for(std::size_t i = 0; i != numPatterns; ++i){
			getBatchElement(outputs,i) = evalPattern(row(patterns,i));
		}
	}
	
	void eval(BatchInputType const& patterns, BatchOutputType & outputs, State& state) const{
		eval(patterns,outputs);
	}
	/// \brief Evaluate the Tree on a single pattern
	void eval(RealVector const& pattern, LabelType& output){
		output = evalPattern(pattern);		
	}

	/// \brief The model does not have any parameters.
	std::size_t numberOfParameters() const{
		return 0;
	}

	/// \brief The model does not have any parameters.
	RealVector parameterVector() const {
		return RealVector();
	}

	/// \brief The model does not have any parameters.
	void setParameterVector(RealVector const& param) {
		SHARK_ASSERT(param.size() == 0);
	}

	/// from ISerializable, reads a model from an archive
	void read(InArchive& archive){
		archive >> m_tree;
		archive >> m_labels;
		archive >> m_inputDimension;
		archive >> m_outputShape;
	}

	/// from ISerializable, writes a model to an archive
	void write(OutArchive& archive) const {
		archive << m_tree;
		archive << m_labels;
		archive << m_inputDimension;
		archive << m_outputShape;
	}

	//Count how often attributes are used
	UIntVector countAttributes() const {
		SHARK_ASSERT(m_inputDimension > 0);
		UIntVector r(m_inputDimension, 0);
		for(auto it = m_tree.begin(); it != m_tree.end(); ++it) {
			if(it->leftId != 0) { // not a label 
				r(it->attributeIndex)++;
			}
		}
		return r;
	}

	///Return input dimension
	Shape inputShape() const {
		return m_inputDimension;
	}
	Shape outputShape() const{
		return m_outputShape;
	}
	
	////////////////////////////////
	/////Tree Construction routines
	///////////////////////////////
	
	std::size_t numberOfNodes() const{
		return m_tree.size();
	}
	
	/// \brief Returns the node with id nodeId
	Node& getNode(std::size_t nodeId){
		SIZE_CHECK(nodeId < m_tree.size());
		return m_tree[nodeId];
	}
	/// \brief Returns the node with id nodeId
	Node const& getNode(std::size_t nodeId)const{
		SIZE_CHECK(nodeId < m_tree.size());
		return m_tree[nodeId];
	}
	
	LabelType const& getLabel(std::size_t nodeId)const{
		SIZE_CHECK(nodeId < m_tree.size());
		return m_labels[m_tree[nodeId].rightIdOrIndex];
	}
	
	/// \brief Creates and returns an untyped root node (neither internal, nor leaf node)
	Node& createRoot(){
		m_tree.clear();
		Node root;
		root.leftId = 0;
		root.rightIdOrIndex = 0;
		m_tree.push_back(root);
		return m_tree[0];
	}
	
	
	///\brief Transforms an untyped node (no child, no internal node) into an internal node
	///
	/// This creates already the two childs of the node, which are untyped.
	Node& transformInternalNode(std::size_t nodeId, std::size_t attributeIndex, double attributeValue) {
		// ids for new child nodes
		int nodeIdLeft = m_tree.size();
		int nodeIdRight = m_tree.size() + 1;
		
		//create new child nodes
		Node leftChild;
		leftChild.leftId = 0;
		leftChild.rightIdOrIndex = 0;
		
		Node rightChild;
		rightChild.leftId = 0;
		rightChild.rightIdOrIndex = 0;
		
		m_tree.push_back(leftChild);
		m_tree.push_back(rightChild);
		
		// connect the parent node with its two childs
		m_tree[nodeId].leftId = nodeIdLeft;
		m_tree[nodeId].rightIdOrIndex = nodeIdRight;
		m_tree[nodeId].attributeIndex = attributeIndex;
		m_tree[nodeId].attributeValue = attributeValue;
		
		return m_tree[nodeId];
	}
	///\brief Transforms a node (no leaf) into a leaf node and inserts the appropriate label
	///
	/// If the node was an internal node before, its connections get removed and the childs
	/// are not reachable any more. Calling a reorder routine like reorderBFS() will get rid of those
	/// nodes.
	Node& transformLeafNode(std::size_t nodeId, LabelType const& label){
		Node& node = m_tree[nodeId];
		node.attributeIndex = 0;
		node. attributeValue = 0.0;
		node.leftId = 0;
		node.rightIdOrIndex = m_labels.size();
		m_labels.push_back(label);
		return node;
	}
	
	/// \brief Reorders a tree into a breath-first-ordering
	///
	/// This function call will remove all unreachable subtrees while reordering
	/// the nodes by their depth in the tree, i.e. first comes the root, the the children
	/// of the root, than their children, etc.
	void reorderBFS(){
		TreeType reordered_tree;
		reordered_tree.reserve(m_tree.size());
		
		std::deque<std::size_t > bfs_queue;
		bfs_queue.push_back(0);
		
		std::size_t nodeId = 0; //running id of the next node to insert
		while(!bfs_queue.empty()){
			Node const& node = getNode(bfs_queue.front());
			bfs_queue.pop_front();
			
			//check leaf
			if(!node.leftId == 0){
				reordered_tree.push_back(node);
			}else{
				reordered_tree.push_back(node);
				reordered_tree.back().leftId = nodeId+1;
				reordered_tree.back().rightIdOrIndex = nodeId+2;
				nodeId += 2;
				bfs_queue.push_back(node.leftId);
				bfs_queue.push_back(node.rightIdOrIndex);
			}
		}
		//overwrite old tree with pruned tree
		m_tree = std::move(reordered_tree);
	}

	/// Find the leaf of the tree for a sample
	template<class Vector>
	std::size_t findLeaf(Vector const& pattern) const{
		std::size_t nodeId = 0;
		while(m_tree[nodeId].leftId != 0){
			if(pattern[m_tree[nodeId].attributeIndex] <= m_tree[nodeId].attributeValue){
				//Branch on left node
				nodeId = m_tree[nodeId].leftId;
			}else{
				//Branch on right node
				nodeId = m_tree[nodeId].rightIdOrIndex;
			}
		}
		return nodeId;
	}

private:
	/// tree of the model
	TreeType m_tree;
	std::vector<LabelType> m_labels;
	///Number of attributes (set by trainer)
	std::size_t m_inputDimension;
	Shape m_outputShape;
	
	/// Evaluate the CART tree on a single sample
	template<class Vector>
	LabelType const& evalPattern(Vector const& pattern) const{
		auto nodeId = findLeaf(pattern);
		return m_labels[m_tree[nodeId].rightIdOrIndex];
	}
};


}
#endif
