//===========================================================================
/*!
 * 
 *
 * \brief       Random Forest Trainer
 * 
 * 
 *
 * \author      K. N. Hansen
 * \date        2011-2012
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
#define SHARK_COMPILE_DLL

#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Models/Trees/RFClassifier.h>
#include <shark/Data/DataView.h>
#include <set>
#include <unordered_map>
#include <shark/Core/OpenMP.h>

using namespace shark;
using namespace std;


//Constructor
RFTrainer::RFTrainer(bool computeFeatureImportances, bool computeOOBerror){
	m_try = 0;
	m_B = 0;
	m_nodeSize = 0;
	m_OOBratio = 0;
	m_regressionLearner = false;
	m_computeFeatureImportances = computeFeatureImportances;
	m_computeOOBerror = computeOOBerror;
}

//Set trainer parameters to sensible defaults
void RFTrainer::setDefaults(){
	if(!m_try){
		if(m_regressionLearner){
			setMTry(static_cast<std::size_t>(std::ceil(m_inputDimension/3.0)));
		}else{
			setMTry(static_cast<std::size_t>(std::ceil(std::sqrt((double)m_inputDimension))));
		}
	}

	if(!m_B){
		setNTrees(100);
	}

	if(!m_nodeSize){
		if(m_regressionLearner){
			setNodeSize(5);
		}else{
			setNodeSize(1);
		}
	}

	if(m_OOBratio <= 0 || m_OOBratio>1){
		setOOBratio(0.66);
	}
}

// Regression
void RFTrainer::train(RFClassifier& model, RegressionDataset const& dataset)
{
	model.clearModels();   // added by TG 23.02.2015

	//TODO O.K.: i am just fixing these things for now so that they are working.

	//Store the number of input dimensions
	m_inputDimension = inputDimension(dataset);

	//Store the size of the labels
	m_labelDimension = labelDimension(dataset);

	model.setInputDimension(m_inputDimension);
	model.setLabelDimension(m_labelDimension);
	auto const n_elements = dataset.numberOfElements();

	m_regressionLearner = true;
	setDefaults();
	
	//we need direct element access since we need to generate elementwise subsets
	std::size_t subsetSize = static_cast<std::size_t>(n_elements*m_OOBratio);
	DataView<RegressionDataset const> elements(dataset);

	auto seed = static_cast<unsigned>(Rng::discrete(0,(unsigned)-1));

	//Generate m_B trees
	SHARK_PARALLEL_FOR(std::uint32_t b = 0; b < m_B; ++b){
		Rng::rng_type rng{seed + b};
		//For each tree generate a subset of the dataset
		//generate indices of the dataset (pick k out of n elements)
		std::vector<std::size_t> trainIndices(n_elements);
		std::iota(trainIndices.begin(),trainIndices.end(),0);
		std::random_shuffle(trainIndices.begin(),trainIndices.end(),DiscreteUniform<>{rng});

		// create oob indices
		auto oobStart = trainIndices.begin() + subsetSize;
		auto oobEnd   = trainIndices.end();

		auto oobIndices = std::vector<std::size_t>(oobStart, oobEnd);
		
		//generate the dataset by copying (TODO: this is a quick fix!
		trainIndices.erase(oobStart, oobEnd);
		auto trainDataView = subset(elements,trainIndices);

		//Create attribute tables
		AttributeTables tables;
		createAttributeTables(trainDataView, tables);

		auto n_trainData = trainDataView.size();
		auto pickLabel = [&](std::size_t i){return trainDataView[i].label;};
		auto labelAvg = detail::cart::sum<RealVector>(n_trainData, pickLabel) / n_trainData;

		TreeType tree = buildTree(tables, trainDataView, labelAvg, 0, rng);
		CARTType cart(tree, m_inputDimension);

		// if oob error or importances have to be computed, create an oob sample
		if(m_computeOOBerror || m_computeFeatureImportances){
			RegressionDataset dataOOB = toDataset(subset(elements, oobIndices));

			// if importances should be computed, oob errors are computed implicitly
			if(m_computeFeatureImportances){
				cart.computeFeatureImportances(dataOOB, rng);
			} // if importances should not be computed, only compute the oob errors
			else{
				cart.computeOOBerror(dataOOB);
			}
		}

		SHARK_CRITICAL_REGION{
			model.addModel(cart);
		}
	}

	if(m_computeOOBerror){
		model.computeOOBerror();
	}

	if(m_computeFeatureImportances){
		model.computeFeatureImportances();
	}
}

// Classification
void RFTrainer::train(RFClassifier& model, ClassificationDataset const& dataset)
{
	model.clearModels();

	m_inputDimension = inputDimension(dataset);
	model.setInputDimension(m_inputDimension);
	m_labelCardinality = numberOfClasses(dataset);
	auto n_elements = dataset.numberOfElements();

	m_regressionLearner = false;
	setDefaults();

	//we need direct element access since we need to generate element-wise subsets
	std::size_t subsetSize = static_cast<std::size_t>(n_elements*m_OOBratio);
	DataView<ClassificationDataset const> elements(dataset);

	auto seed = static_cast<unsigned>(Rng::discrete(0,(unsigned)-1));

	auto oobClassTally = UIntMatrix{n_elements,m_labelCardinality};

	//Generate m_B trees
	SHARK_PARALLEL_FOR(std::uint32_t b = 0; b < m_B; ++b){
		Rng::rng_type rng{seed + b};
		//For each tree generate a subset of the dataset
		//generate indices of the dataset (pick k out of n elements)
		std::vector<std::size_t> trainIndices(n_elements);
		std::iota(trainIndices.begin(),trainIndices.end(),0);
		std::random_shuffle(trainIndices.begin(),trainIndices.end(),DiscreteUniform<>{rng});

		// create oob indices
		auto oobStart = trainIndices.begin() + subsetSize;
		auto oobEnd   = trainIndices.end();

		auto oobIndices = std::vector<std::size_t>(oobStart, oobEnd);

		//generate the dataset by copying (TODO: this is a quick fix!
		trainIndices.erase(oobStart, oobEnd);
		auto trainDataView = subset(elements,trainIndices);

		//Create attribute tables
		AttributeTables tables;
		createAttributeTables(trainDataView, tables);
		auto cAbove = createCountVector(trainDataView);

		TreeType tree = buildTree(tables, trainDataView, cAbove, 0, rng);
		CARTType cart(tree, m_inputDimension);

		// if oob error or importances have to be computed, create an oob sample
		if(m_computeOOBerror || m_computeFeatureImportances){
			ClassificationDataset dataOOB = toDataset(subset(elements, oobIndices));

			// if importances should be computed, oob errors are computed implicitly
			if(m_computeFeatureImportances){
				cart.computeFeatureImportances(dataOOB, rng);
			} // if importances should not be computed, only compute the oob errors
			else{
				cart.computeOOBerror(dataOOB);
			}
		}

		SHARK_CRITICAL_REGION{
			model.addModel(cart);
			for(auto i : oobIndices){
				auto histogram = cart(elements[i].input);
				auto j = detail::cart::argmax(histogram);
				++oobClassTally(i,j);
			}
		}
	}

	// compute the oob error for the whole ensemble
	if(m_computeOOBerror){
		model.computeOOBerror(oobClassTally,elements);
	}

	// compute the feature importances for the whole ensemble
	if(m_computeFeatureImportances){
		model.computeFeatureImportances();
	}
}

void RFTrainer::setMTry(std::size_t mtry){
	m_try = mtry;
}

void RFTrainer::setNTrees(std::size_t nTrees){
	m_B = nTrees;
}

void RFTrainer::setNodeSize(std::size_t nodeSize){
	m_nodeSize = nodeSize;
}

void RFTrainer::setOOBratio(double ratio){
	m_OOBratio = ratio;
}



TreeType RFTrainer::
buildTree(AttributeTables& tables,
		  DataView<ClassificationDataset const> const& elements,
		  ClassVector& cAbove, std::size_t nodeId,
		  Rng::rng_type& rng){

	//Construct tree
	TreeType lTree, rTree, tree;
	NodeInfo nodeInfo{nodeId};

	//n = Total number of cases in the dataset
	std::size_t n = tables[0].size();

	bool isLeaf = false;
	if(gini(cAbove,tables[0].size())==0 || n <= m_nodeSize){
		isLeaf = true;
	}else{
		//Count vectors
		ClassVector cBelow(m_labelCardinality),cBestBelow(m_labelCardinality),
				cBestAbove(m_labelCardinality);

		//Randomly select the attributes to test for split
		auto tableIndices = generateRandomTableIndices(rng);

		//Index of attributes
		std::size_t bestAttributeValIndex = 0;

		//Attribute values
		constexpr double worstImpurity = std::numeric_limits<double>::max();
		double bestImpurity = worstImpurity; //old: n+1.0;

		for (std::size_t attributeIndex : tableIndices){
			auto const& attributeTable = tables[attributeIndex];
			auto cTmpAbove = cAbove;
			std::fill(cBelow.begin(),cBelow.end(),0);
			for(std::size_t prev=0,i=1; i<n; prev=i++){
				auto const& label = elements[attributeTable[prev].id].label;
				// Update the count of the label
				++cBelow[label];
				--cTmpAbove[label];
				if(attributeTable[prev].value == attributeTable[i].value) continue;

				// n1/n2 = Number of cases to the left/right of child node
				std::size_t n1 = i,    n2 = n-i;

				//Calculate the Gini impurity of the split
				double impurity = n1*gini(cBelow,n1)+n2*gini(cTmpAbove,n2);
				if(impurity<bestImpurity){
					//Found a more pure split, store the attribute index and value
					nodeInfo.attributeIndex = attributeIndex;
					nodeInfo.attributeValue = attributeTable[prev].value;
					bestAttributeValIndex = prev;
					bestImpurity = impurity;
					cBestAbove = cTmpAbove;
					cBestBelow = cBelow;
				}
			}
		}

		if(bestImpurity<worstImpurity){
			AttributeTables rTables, lTables;
			splitAttributeTables(tables, nodeInfo.attributeIndex, bestAttributeValIndex, lTables, rTables);
			tables.clear();
			//Continue recursively

			nodeInfo.leftNodeId  = (nodeId<<1)+1;
			nodeInfo.rightNodeId = (nodeId<<1)+2;

			lTree = buildTree(lTables, elements, cBestBelow, nodeInfo.leftNodeId, rng);
			rTree = buildTree(rTables, elements, cBestAbove, nodeInfo.rightNodeId, rng);
		}else{
			//Leaf node
			isLeaf = true;
		}

	}

	//Store entry in the tree table
	if(isLeaf){
		nodeInfo.label = hist(cAbove);
		tree.push_back(std::move(nodeInfo));
		return tree;
	}

	tree.push_back(std::move(nodeInfo));
	tree.insert(tree.end(), lTree.begin(), lTree.end());
	tree.insert(tree.end(), rTree.begin(), rTree.end());

	return tree;
}

RFTrainer::LabelType RFTrainer::hist(ClassVector const& countVector) const {

	LabelType histogram(m_labelCardinality,0.0);

	std::size_t totalElements = 0;

	for (std::size_t i = 0, s = countVector.size(); i<s; ++i){
		histogram(i) = countVector[i];
		totalElements += countVector[i];
	}
	histogram /= totalElements;

	return histogram;
}

TreeType RFTrainer::
buildTree(AttributeTables& tables,
		  DataView<RegressionDataset const> const& elements,
		  LabelType const& labelAvg,
		  std::size_t nodeId, Rng::rng_type& rng){

	//Construct tree
	TreeType lTree, rTree, tree;
	NodeInfo nodeInfo{nodeId,labelAvg};

	//n = Total number of cases in the dataset
	std::size_t n = tables[0].size();

	bool isLeaf = false;
	if(n <= m_nodeSize){
		isLeaf = true;
	}else{
		//label vectors
		LabelVector tmpLabels;
		LabelType bestAvgAbove,bestAvgBelow;
		RealVector labelSumAbove(m_labelDimension), labelSumBelow(m_labelDimension);

		//Randomly select the attributes to test for split
		auto tableIndices = generateRandomTableIndices(rng);

		//Index of attributes
		std::size_t bestAttributeValIndex = 0;

		//Attribute values
		constexpr double worstImpurity = std::numeric_limits<double>::max();
		double bestImpurity = worstImpurity;

		for (std::size_t attributeIndex : tableIndices){
			auto const& attributeTable = tables[attributeIndex];

			labelSumBelow.clear();
			labelSumAbove.clear();
			tmpLabels.clear();

			//Create a labels table, that corresponds to the sorted attribute
			for(auto const& el : attributeTable){
				auto const& label = elements[el.id].label;
				tmpLabels.push_back(label);
				labelSumBelow += label;
			}

			for(std::size_t prev=0,i=1; i<n; prev=i++){
				labelSumAbove += tmpLabels[prev];
				labelSumBelow -= tmpLabels[prev];
				if(attributeTable[prev].value == attributeTable[i].value) continue;

				std::size_t n1=i,    n2 = n-i;

				//Calculate the squared error of the split
				auto avgAbove = labelSumAbove/n1,   avgBelow = labelSumBelow/n2;
				double impurity = (n1*totalSumOfSquares(tmpLabels,0,n1,avgAbove)
								   +n2*totalSumOfSquares(tmpLabels,n1,n2,avgBelow))/n;
				if(impurity<bestImpurity){
					//Found a more pure split, store the attribute index and value
					nodeInfo.attributeIndex = attributeIndex;
					nodeInfo.attributeValue = attributeTable[prev].value;
					bestAttributeValIndex = prev;
					bestImpurity = impurity;
					bestAvgAbove = avgAbove;
					bestAvgBelow = avgBelow;
				}
			}
		}

		if(bestImpurity<worstImpurity){
			//Split the attribute tables
			AttributeTables rTables, lTables;
			splitAttributeTables(tables, nodeInfo.attributeIndex, bestAttributeValIndex, lTables, rTables);
			tables.clear();//save memory

			//Continue recursively
			nodeInfo.leftNodeId = (nodeId<<1)+1;
			nodeInfo.rightNodeId = (nodeId<<1)+2;

			lTree = buildTree(lTables, elements, bestAvgAbove, nodeInfo.leftNodeId, rng);
			rTree = buildTree(rTables, elements, bestAvgBelow, nodeInfo.rightNodeId, rng);
		}else{
			//Leaf node
			isLeaf = true;
		}

	}

	if(isLeaf){
		tree.push_back(std::move(nodeInfo));
		return tree;
	}

	tree.push_back(std::move(nodeInfo));
	tree.insert(tree.end(), lTree.begin(), lTree.end());
	tree.insert(tree.end(), rTree.begin(), rTree.end());

	//Store entry in the tree
	return tree;

}



/**
 * Returns the average vector of a vector of real vectors
 */
RealVector RFTrainer::average(LabelVector const& labels) const {
	RealVector avg(labels[0].size());
	for(auto const& label : labels) avg += label;
	return avg/labels.size();
}

double RFTrainer::totalSumOfSquares(
		LabelVector const& labels,
		std::size_t start, std::size_t length,
		LabelType const& labelAvg) const{
	if (length < 1)
		throw SHARKEXCEPTION("[RFTrainer::totalSumOfSquares] length < 1");
	if (start+length > labels.size())
		throw SHARKEXCEPTION("[RFTrainer::totalSumOfSquares] start+length > labels.size()");

	return detail::cart::sum<double>(start, start+length,
					   [&](std::size_t const i){
						   return distanceSqr(labels[i],labelAvg);
					   });
}

/**
 * Returns two attribute tables: LAttrbuteTables and RAttrbuteTables
 * Calculated from splitting tables at (index, valIndex)
 */
void RFTrainer::splitAttributeTables(AttributeTables const& tables, std::size_t index, std::size_t valIndex, AttributeTables& LAttributeTables, AttributeTables& RAttributeTables){
	AttributeTable table;

	//Build a hash table for fast lookup
	std::unordered_map<std::size_t, bool> hash;
	for(std::size_t i = 0; i< tables[index].size(); i++){
		hash[tables[index][i].id] = (i<=valIndex);
	}

	for(std::size_t i = 0; i < tables.size(); i++){
		//For each attribute table
		LAttributeTables.push_back(table);
		RAttributeTables.push_back(table);
		for(std::size_t j = 0; j < tables[i].size(); j++){
			if(hash[tables[i][j].id]){
				//Left
				LAttributeTables[i].push_back(tables[i][j]);
			}else{
				//Right
				RAttributeTables[i].push_back(tables[i][j]);
			}
		}
	}
}


///Generates a random set of indices
set<std::size_t> RFTrainer::generateRandomTableIndices(Rng::rng_type& rng) const {
	set<std::size_t> tableIndices;
	DiscreteUniform<> discrete{rng,0,m_inputDimension-1};
	//Draw the m_try Generate the random attributes to search for the split
	while(tableIndices.size()<m_try){
		tableIndices.insert(discrete());
	}
	return tableIndices;
}

///Calculates the Gini impurity of a node. The impurity is defined as
///1-sum_j p(j|t)^2
///i.e the 1 minus the sum of the squared probability of observing class j in node t
double RFTrainer::gini(ClassVector const& countVector, std::size_t n) const{
	if(!n) return 1;

	double res = 0.;
	n *= n;
	for(auto const& i: countVector){
		res += sqr(i)/ static_cast<double>(n);
	}
	return 1-res;
}


/// Creates the attribute tables, and in the process creates a count vector (cAbove).
/// A dataset consisting of m input variables has m attribute tables.
/// [attribute | class/value | rid ]
template<class Dataset>
void RFTrainer::createAttributeTables(DataView<Dataset const> const& elements, AttributeTables& tables){
	std::size_t n_elements = elements.size();
	//Each entry in the outer vector is an attribute table
	AttributeTable table;
	//For each column
	for(std::size_t j=0; j<m_inputDimension; j++){
		table.clear();
		//For each row, store Attribute value, class and rowId
		for(std::size_t i=0; i<n_elements; i++){
			table.push_back(RFAttribute{elements[i].input[j], i});
		}
		std::sort(table.begin(), table.end(), tableSort);
		//Store this attributes attribute table
		tables.push_back(table);
	}
}


RFTrainer::ClassVector RFTrainer::
createCountVector(DataView<ClassificationDataset const> const& elements) const {
	ClassVector cAbove = ClassVector(m_labelCardinality);
	for(std::size_t i = 0, s = elements.size(); i<s; ++i){
		++cAbove[elements[i].label];
	}
	return cAbove;
}

bool RFTrainer::tableSort(RFAttribute const& v1, RFAttribute const& v2) {
	return v1.value < v2.value;
}
