//===========================================================================
/*!
 * 
 *
 * \brief       CART
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


#ifndef SHARK_ALGORITHMS_TRAINERS_CARTTRAINER_H
#define SHARK_ALGORITHMS_TRAINERS_CARTTRAINER_H


#include <shark/Models/Trees/CARTClassifier.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <boost/unordered_map.hpp>

namespace shark {
/*!
 * \brief Classification And Regression Trees CART
 *
 * CART is a decision tree algorithm, that builds a binary decision tree
 * The decision tree is built by partitioning a dataset recursively
 *
 * The partitioning is done, so that the partition chosen at a single
 * node, is the partition the produces the largest decrease in node
 * impurity.
 *
 * The node impurity is measured by the Gini criteria in the classification
 * case, and the total sum of squares error in the regression case
 *
 * The tree is grown, until all leafs are pure. A node is considered pure
 * when it only consist of identical cases in the classification case
 * and identical or single values in the regression case
 *
 * After the maximum sized tree is grown, the tree is pruned back from the leafs upward.
 * The pruning is done by cost complexity pruning, as described by L. Breiman
 *
 * The algorithm used is based on the SPRINT algorithm, as shown by J. Shafer et al.
 *
 * For more detailed information about CART, see \e Classification \e And \e Regression
 * \e Trees written by L. Breiman et al. 1984.
 */
class CARTTrainer 
: public AbstractTrainer<CARTClassifier<RealVector>, unsigned int>
, public AbstractTrainer<CARTClassifier<RealVector>, RealVector >
{
public:
    typedef CARTClassifier<RealVector> ModelType;

	/// Constructor
	CARTTrainer(){
		m_nodeSize = 1;
		m_numberOfFolds = 10;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CARTTrainer"; }

	///Train classification
	void train(ModelType& model, ClassificationDataset const& dataset);
	
	///Train regression
	void train(ModelType& model, RegressionDataset const& dataset);

	///Sets the number of folds used for creation of the trees.
	void setNumberOfFolds(unsigned int folds){
		m_numberOfFolds = folds;
	}
protected:

	///Types frequently used
	struct TableEntry{
		double value;
		std::size_t id;

		bool operator<( TableEntry const& v2)const {
			return value < v2.value;
		}
	};
	typedef std::vector < TableEntry > AttributeTable;
	typedef std::vector < AttributeTable > AttributeTables;

	typedef ModelType::SplitMatrixType SplitMatrixType;


	///Number of attributes in the dataset
	std::size_t m_inputDimension;

	///Size of labels
	std::size_t m_labelDimension;

	///Controls the number of samples in the terminal nodes
	std::size_t m_nodeSize;

	///Holds the maximum label. Used in allocating the histograms
	unsigned int m_maxLabel;

	///Number of folds used to create the tree.
	unsigned int m_numberOfFolds;

	//Classification functions
	///Builds a single decision tree from a classification dataset
	///The method requires the attribute tables,
	SplitMatrixType buildTree(AttributeTables const& tables, ClassificationDataset const& dataset, boost::unordered_map<std::size_t, std::size_t>& cAbove, std::size_t nodeId );

	///Calculates the Gini impurity of a node. The impurity is defined as
	///1-sum_j p(j|t)^2
	///i.e the 1 minus the sum of the squared probability of observing class j in node t
	double gini(boost::unordered_map<std::size_t, std::size_t>& countMatrix, std::size_t n);
	///Creates a histogram from the count matrix.
	RealVector hist(boost::unordered_map<std::size_t, std::size_t> countMatrix);

	///Regression functions
	SplitMatrixType buildTree(AttributeTables const& tables, RegressionDataset const& dataset, std::vector<RealVector> const& labels, std::size_t nodeId, std::size_t trainSize);
	///Calculates the total sum of squares
	double totalSumOfSquares(std::vector<RealVector> const& labels, std::size_t start, std::size_t length, const RealVector& sumLabel);
	///Calculates the mean of a vector of labels
	RealVector mean(std::vector<RealVector> const& labels);

	///Pruning
	///Prunes decision tree, represented by a split matrix
	void pruneMatrix(SplitMatrixType& splitMatrix);
	///Prunes a single node, including the child nodes of the decision tree
	void pruneNode(SplitMatrixType& splitMatrix, std::size_t nodeId);
	///Updates the node variables used in the cost complexity pruning stage
	void measureStrenght(SplitMatrixType& splitMatrix, std::size_t nodeId, std::size_t parentNodeId);

	///Returns the index of the node with node id in splitMatrix.
	std::size_t findNode(SplitMatrixType& splitMatrix, std::size_t nodeId);

	///Attribute table functions
	///Create the attribute tables used by the SPRINT algorithm
	AttributeTables createAttributeTables(Data<RealVector> const& dataset);
	///Splits the attribute tables by a attribute index and value. Returns a left and a right attribute table in the variables LAttributeTables and RAttributeTables
	void splitAttributeTables(AttributeTables const& tables, std::size_t index, std::size_t valIndex, AttributeTables& LAttributeTables, AttributeTables& RAttributeTables);
	///Crates count matrices from a classification dataset
	boost::unordered_map<std::size_t, std::size_t> createCountMatrix(ClassificationDataset const& dataset);


};


}
#endif

