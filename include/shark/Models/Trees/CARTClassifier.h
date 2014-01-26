//===========================================================================
/*!
 * 
 *
 * \brief       Cart Classifier
 * 
 * 
 *
 * \author      K. N. Hansen
 * \date        2012
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

#ifndef SHARK_MODELS_TREES_CARTCLASSIFIER_H
#define SHARK_MODELS_TREES_CARTCLASSIFIER_H


#include <shark/Models/AbstractModel.h>                        
#include <shark/Data/Dataset.h>

namespace shark {


///
/// \brief CART Classifier.
///
/// \par
/// The CARTClassifier predicts a class label
/// using the CART algorithm.
///
/// \par
/// It is a decision tree algorithm.
///
template<class LabelType>
class CARTClassifier : public AbstractModel<RealVector,LabelType>
{
private:
	typedef AbstractModel<RealVector, LabelType> base_type;
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
//	Information about a single split. misclassProp, r and g are variables used in the cost complexity step
	struct SplitInfo{
		std::size_t nodeId;
		std::size_t attributeIndex;
		double attributeValue;
		std::size_t leftNodeId;
		std::size_t rightNodeId;
		LabelType label;
		double misclassProp;//TODO: remove this
		std::size_t r;//TODO: remove this
		double g;//TODO: remove this

	   template<class Archive>
	   void serialize(Archive & ar, const unsigned int version){
			ar & nodeId;
			ar & attributeIndex;
			ar & attributeValue;
			ar & leftNodeId;
			ar & rightNodeId;
			ar & label;
			ar & misclassProp;
			ar & r;
			ar & g;
		}
	};

	/// Vector of struct's that contains the splitting information and the labels.
	/// The class label is a normalized histogram in the classification case.
	/// In the regression case, the label is the regression value.
	typedef std::vector<SplitInfo> SplitMatrixType;

	/// Constructor
	CARTClassifier()
	{}

	/// Constructor taking the splitMatrix as argument
	CARTClassifier(SplitMatrixType const& splitMatrix)
	{
		setSplitMatrix(splitMatrix);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CARTClassifier"; }

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;
	/// \brief Evaluate the Tree on a batch of patterns
	void eval(const BatchInputType& patterns, BatchOutputType& outputs)const{
		std::size_t numPatterns = shark::size(patterns);
		//evaluate the first pattern alone and create the batch output from that
		LabelType const& firstResult = evalPattern(row(patterns,0));
		outputs = Batch<LabelType>::createBatch(firstResult,numPatterns);
		get(outputs,0) = firstResult;
		
		//evaluate the rest
		for(std::size_t i = 0; i != numPatterns; ++i){
			get(outputs,i) = evalPattern(row(patterns,i));
		}
	}
	
	void eval(const BatchInputType& patterns, BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}
	/// \brief Evaluate the Tree on a single pattern
	void eval(RealVector const & pattern, LabelType& output){
		output = evalPattern(pattern);		
	}

	/// Set the model split matrix.
	void setSplitMatrix(SplitMatrixType const& splitMatrix){
		m_splitMatrix = splitMatrix;
		optimizeSplitMatrix(m_splitMatrix);
	}
	
	/// \brief The model does not have any parameters.
	std::size_t numberOfParameters()const{
		return 0;
	}

	/// \brief The model does not have any parameters.
	RealVector parameterVector() const {
		return RealVector();
	}

	/// \brief The model does not have any parameters.
	void setParameterVector(const RealVector& param) {
		SHARK_ASSERT(param.size() == 0);
	}

	/// from ISerializable, reads a model from an archive
	void read(InArchive& archive){
		archive >> m_splitMatrix;
	}

	/// from ISerializable, writes a model to an archive
	void write(OutArchive& archive) const {
		archive << m_splitMatrix;
	}

protected:
	/// split matrix of the model
	SplitMatrixType m_splitMatrix;
	
	/// \brief Finds the index of the node with a certain nodeID in an unoptimized split matrix.
	std::size_t findNode(std::size_t nodeId)const{
		std::size_t index = 0;
		for(; nodeId != m_splitMatrix[index].nodeId; ++index);
		return index;
	}

	/// Optimize a split matrix, so constant lookup can be used.
	/// The optimization is done by changing the index of the children
	/// to use indices instead of node ID.
	/// Furthermore, the node IDs are converted to index numbers.
	void optimizeSplitMatrix(SplitMatrixType& splitMatrix)const{
		for(std::size_t i = 0; i < splitMatrix.size(); i++){
			splitMatrix[i].leftNodeId = findNode(splitMatrix[i].leftNodeId);
			splitMatrix[i].rightNodeId = findNode(splitMatrix[i].rightNodeId);
		}
		for(std::size_t i = 0; i < splitMatrix.size(); i++){
			splitMatrix[i].nodeId = i;
		}
	}
	
	/// Evaluate the CART tree on a single sample
	template<class Vector>
	LabelType const& evalPattern(Vector const& pattern)const{
		std::size_t nodeId = 0;
		while(m_splitMatrix[nodeId].leftNodeId != 0){
			if(pattern[m_splitMatrix[nodeId].attributeIndex]<=m_splitMatrix[nodeId].attributeValue){
				//Branch on left node
				nodeId = m_splitMatrix[nodeId].leftNodeId;
			}else{
				//Branch on right node
				nodeId = m_splitMatrix[nodeId].rightNodeId;
			}
		}
		return m_splitMatrix[nodeId].label;
	}
};


}
#endif
