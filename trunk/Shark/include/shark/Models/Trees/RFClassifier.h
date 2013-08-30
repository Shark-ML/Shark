//===========================================================================
/*!
*
*  \brief Random Forest Classifier.
*
*  \author  K. N. Hansen, O.Krause
*  \date    2011-2012
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

#ifndef SHARK_MODELS_TREES_RFCLASSIFIER_H
#define SHARK_MODELS_TREES_RFCLASSIFIER_H

#include <shark/Models/Trees/CARTClassifier.h>

//#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
//#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>

//#include <boost/unordered_map.hpp>

namespace shark {


///
/// \brief Random Forest Classifier.
///
/// \par
/// The Random Forest Classifier predicts a class label
/// using the Random Forest algorithm as described in<br/>
/// Random Forests. Leo Breiman. Machine Learning, 1(45), pages 5-32. Springer, 2001.<br/>
///
/// \par
/// It is a ensemble learner that uses multiple decision trees built
/// using the CART methodology.
///
class RFClassifier : public AbstractModel<RealVector, RealVector>
{
public:

	/// Vector of struct's that contains the splitting information and the labels.
	/// The class label is a normalized histogram in the classification case.
	/// In the regression case, the label is the regression value.
	typedef CARTClassifier<RealVector>::SplitMatrixType SplitMatrixType;
	typedef CARTClassifier<RealVector>::SplitInfo SplitInfo;
	
	/// Constructor
	RFClassifier(){
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RFClassifier"; }
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	// Evaluate the random forest model.
	// In the regression case the function returns the average vector.
	using AbstractModel<RealVector, RealVector >::eval;
	void eval(const BatchInputType& patterns, BatchOutputType& outputs)const{
		// Prepare the output
		ensureSize(outputs,size(patterns),m_labelDimension);
		zero(outputs);

		//For each tree in the forest and every input pattern
		for(std::size_t i = 0; i != m_forest.size(); i++){
			noalias(outputs) += m_forest[i](patterns);
		}

		outputs /= m_forest.size();
	}
	
	void eval(const BatchInputType& patterns, BatchOutputType& outputs, State & state)const{
		eval(patterns,outputs);
	}


	/// This model does not have any parameters.
	RealVector parameterVector() const {
		return RealVector();
	}

	/// This model does not have any parameters
	void setParameterVector(const RealVector& param) {
		SHARK_ASSERT(param.size() == 0);
	}

	/// from ISerializable, reads a model from an archive
	void read(InArchive& archive){
		archive >> m_forest;
	}

	/// from ISerializable, writes a model to an archive
	void write(OutArchive& archive)const{
		archive << m_forest;
	}

	/// \brief Adds a new tree to the ensemble.
	///
	/// \param splitMatrix the Tree to add in form of a splitMatrix
	void addTree(SplitMatrixType const& splitMatrix){
		/// Add the tree to the ensemble
		m_forest.push_back(splitMatrix);

//		//TODO: O.K. : Is this needed at all? It is not used.
//		// Update OOB error
//		for(std::size_t i=0; i<testSet.size(); i++){
//			RealVector prediction = m_forest.back()(testSet[i].input);
//			std::size_t index = testSet.index(i);
//			if(m_OOBHashTable.find(index) == m_OOBHashTable.end()){
//				m_OOBHits.push_back(1);
//				m_OOBPrediction.push_back(prediction);
//				m_OOBLabel.push_back(testSet[i].label);
//				m_OOBHashTable[index] = m_OOBLabel.size()-1;
//			}else{
//				std::size_t id = m_OOBHashTable[index];
//				m_OOBPrediction[id] += prediction;
//				m_OOBHits[id]++;
//			}
//		}
	}

	/// Set the dimension of the labels
	void setLabelDimension(std::size_t in){
		m_labelDimension = in;
	}

protected:
	/// collection of trees. Each tree consists of a split matrix.
	std::vector< CARTClassifier<RealVector> > m_forest;

	// Dimension of label in the regression case, number of classes in the classification case.
	std::size_t m_labelDimension;
	
//	//TODO: O.K. : Is this needed at all? It is not used(applies to the remaining stuff)
//	/// Hash table; so constant lookup can be applied when calculating the OOB error
//	boost::unordered_map< std::size_t, std::size_t > m_OOBHashTable;

//	// predicted labels of each node
//	std::vector< RealVector > m_OOBPrediction;

//	/// number of predicted hits at each node
//	std::vector< std::size_t > m_OOBHits;

//	// labels
//	std::vector< RealVector > m_OOBLabel;
};


}
#endif
