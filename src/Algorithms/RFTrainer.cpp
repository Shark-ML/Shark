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
RFTrainer::RFTrainer(bool computeFeatureImportances, bool computeOOBerror)
: m_B{100}, m_OOBratio{0.66}
{
	m_try = 0;
	m_nodeSize = 0;
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
			setMTry(static_cast<std::size_t>(std::ceil(std::sqrt(m_inputDimension))));
		}
	}

	if(!m_nodeSize){
		if(m_regressionLearner) setNodeSize(5);
		else setNodeSize(1);
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
		trainIndices.resize(subsetSize);
		auto trainDataView = subset(elements,trainIndices);

		//Create attribute tables
		AttributeTables tables;
		createAttributeTables(trainDataView, tables);

		auto n_trainData = trainDataView.size();
		auto labelSum = detail::cart::sum<RealVector>(n_trainData, [&](std::size_t i){
			return trainDataView[i].label;
		});

		TreeType tree = buildTree(tables, trainDataView, labelSum, 0, rng);
		//TreeType tree = build(trainDataView,labelAvg,rng);
		CARTType cart(std::move(tree), m_inputDimension);

		// if oob error or importances have to be computed, create an oob sample
		if(m_computeOOBerror || m_computeFeatureImportances){
			RegressionDataset dataOOB = toDataset(subset(elements, oobIndices));

			// cart oob errors are computed implicitly whenever importances are
			if(m_computeFeatureImportances){
				cart.computeFeatureImportances(dataOOB, rng);
			} else cart.computeOOBerror(dataOOB);
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

	UIntMatrix oobClassTally = UIntMatrix(n_elements,m_labelCardinality);

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
		trainIndices.resize(subsetSize);
		auto trainDataView = subset(elements,trainIndices);

		//Create attribute tables
		AttributeTables tables;
		createAttributeTables(trainDataView, tables);
		auto cAbove = createCountVector(trainDataView);

		TreeType tree = buildTree(tables, trainDataView, cAbove, 0, rng);
		CARTType cart(std::move(tree), m_inputDimension);

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
			for(auto const i : oobIndices){
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

TreeType RFTrainer::
buildTree(detail::cart::sink<AttributeTables&> tables,
		  DataView<ClassificationDataset const> const& elements,
		  ClassVector& cFull, std::size_t nodeId,
		  Rng::rng_type& rng){

	//Construct tree
	TreeType lTree, rTree, tree;
	tree.push_back(NodeInfo{nodeId});
	NodeInfo& nodeInfo = tree[0];

	//n = Total number of cases in the dataset
	std::size_t n = tables[0].size();
	if(gini(cFull,n)==0.0 || n <= m_nodeSize) {
		nodeInfo.label = hist(cFull);
		return tree;
	}

	//Randomly select the attributes to test for split
	auto tableIndices = generateRandomTableIndices(rng);

	auto split = findSplit(tables,elements,cFull,tableIndices);

	// if the purity hasn't improved, this is a leaf.
	if(split.impurity==WORST_IMPURITY) {
		// this shouldn't really happen: leaves ought to be cought by previous check
		// TODO(jwrigley): Why is hist only applied to leaves, when average is applied to all nodes
		nodeInfo.label = hist(cFull);
		return tree;
	}
	nodeInfo <<= split;
	AttributeTables rTables, lTables;
	splitAttributeTables(tables, split.splitAttribute, split.splitRow, lTables, rTables);
	//Continue recursively

	nodeInfo.leftNodeId  = (nodeId<<1)+1;   nodeInfo.rightNodeId = (nodeId<<1)+2;

	lTree = buildTree(lTables, elements, split.cBelow, nodeInfo.leftNodeId, rng);
	rTree = buildTree(rTables, elements, split.cAbove, nodeInfo.rightNodeId, rng);

//	tree.insert(tree.end(), std::make_move_iterator(lTree.begin()), std::make_move_iterator(lTree.end()));
//	tree.insert(tree.end(), std::make_move_iterator(rTree.begin()), std::make_move_iterator(rTree.end()));
	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(),lTree.end(),std::back_inserter(tree));
	std::move(rTree.begin(),rTree.end(),std::back_inserter(tree));
	return tree;
}

RFTrainer::Split RFTrainer::findSplit(
		RFTrainer::AttributeTables const& tables,
		DataView<ClassificationDataset const> const& elements,
		ClassVector const& cFull,
		set<size_t> const& tableIndices) const
{
	auto n = tables[0].size();
	Split best;
	for (std::size_t attributeIndex : tableIndices){
		auto const& attributeTable = tables[attributeIndex];
		auto cAbove = cFull;
		auto cBelow = ClassVector(m_labelCardinality);
		std::fill(cBelow.begin(),cBelow.end(),0);
		for(std::size_t prev=0,i=1; i<n; prev=i++){
			auto const& label = elements[attributeTable[prev].id].label;
			// Pass the label down
			--cAbove[label];    ++cBelow[label];
			if(attributeTable[prev].value == attributeTable[i].value) continue;

			// n1/n2 = Number of cases to the left/right of child node
			std::size_t n1 = i,    n2 = n-i;

			//Calculate the Gini impurity of the split
			double impurity = n1*gini(cBelow,n1)+n2*gini(cAbove,n2);
			if(impurity<best.impurity){
				//Found a more pure split, store the attribute index and value
				best.splitAttribute = attributeIndex;
				best.splitRow = prev;
				best.splitValue = attributeTable[prev].value;
				best.impurity = impurity;
				best.cAbove = cAbove;
				best.cBelow = cBelow;
			}
		}
	}
	return best;
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
buildTree(detail::cart::sink<AttributeTables&> tables,
		  DataView<RegressionDataset const> const& elements,
		  LabelType const& sumFull,
		  std::size_t nodeId, Rng::rng_type& rng){

	//Construct tree
	TreeType lTree, rTree, tree;
	auto n = tables[0].size();
	// TODO(jwrigley): Why is average assigned to all nodes, when hist is only applied to leaves?
	tree.push_back(NodeInfo{nodeId,sumFull/n});
	NodeInfo& nodeInfo = tree[0];

	//n = Total number of cases in the dataset
	if(n <= m_nodeSize) return tree; // Must be leaf

	//Randomly select the attributes to test for split
	auto tableIndices = generateRandomTableIndices(rng);

	auto split = findSplit(tables, elements, sumFull,tableIndices);

	// if the purity hasn't improved, this is a leaf.
	//if(split.impurity==WORST_IMPURITY) return tree;
	if(split.purity == 0) return tree;
	nodeInfo <<= split;

	//Split the attribute tables
	AttributeTables rTables, lTables;
	splitAttributeTables(tables, split.splitAttribute, split.splitRow, lTables, rTables);

	//Continue recursively
	nodeInfo.leftNodeId = (nodeId<<1)+1;
	nodeInfo.rightNodeId = (nodeId<<1)+2;

	lTree = buildTree(lTables, elements, split.sumAbove, nodeInfo.leftNodeId, rng);
	rTree = buildTree(rTables, elements, split.sumBelow, nodeInfo.rightNodeId, rng);

//	tree.insert(tree.end(), std::make_move_iterator(lTree.begin()), std::make_move_iterator(lTree.end()));
//	tree.insert(tree.end(), std::make_move_iterator(rTree.begin()), std::make_move_iterator(rTree.end()));
	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(),lTree.end(),std::back_inserter(tree));
	std::move(rTree.begin(),rTree.end(),std::back_inserter(tree));
	return tree;
}

RFTrainer::Split RFTrainer::findSplit (
		RFTrainer::AttributeTables const &tables,
		DataView<RegressionDataset const> const &elements,
		RealVector const& sumFull,
		set<size_t> const &tableIndices) const
{
	auto n = tables[0].size();
	Split best{};
	LabelType const sumEmpty(m_labelDimension,0);
	LabelVector tmp(n);
	for (std::size_t const attributeIndex : tableIndices){
		auto const& attributeTable = tables[attributeIndex];
		auto sumBelow = sumFull, sumAbove = sumEmpty;
		//Create a labels table, that corresponds to the sorted attribute
		detail::cart::fill_fn(tmp,[&](std::size_t i){
			return elements[attributeTable[i].id].label;
		});

		for(std::size_t prev=0,i=1; i<n; prev=i++){
			sumAbove += tmp[prev]; sumBelow -= tmp[prev];
			if(attributeTable[prev].value == attributeTable[i].value) continue;

			std::size_t n1=i,    n2 = n-i;
			//Calculate the squared error of the split
			double sumSqAbove = norm_sqr(sumAbove), sumSqBelow = norm_sqr(sumBelow);
			double purity =  sumSqAbove/n1 + sumSqBelow/n2;
			if(purity>best.purity){
				//Found a more pure split, store the attribute index and value
				best.splitAttribute = attributeIndex;
				best.splitRow = prev;
				best.splitValue = (attributeTable[prev].value + attributeTable[i].value)/2.0;
				// Somewhat faster, but probably less accurate
				//best.splitValue = attributeTable[prev].value;
				best.purity = purity;
				best.sumAbove = sumAbove;
				best.sumBelow = sumBelow;
			}
		}
	}
	return best;
}

/**
 * Returns two attribute tables: LAttrbuteTables and RAttrbuteTables
 * Calculated from splitting tables at (index, valIndex)
 */
void RFTrainer::splitAttributeTables(
		detail::cart::sink<AttributeTables &> tables,
		std::size_t index, std::size_t valIndex,
		AttributeTables& LAttributeTables,
		AttributeTables& RAttributeTables)
{

	//Build a hash table for fast lookup
	std::unordered_map<std::size_t, bool> hash;
	for(std::size_t i = 0, s = tables[index].size(); i<s; ++i) {
		hash[tables[index][i].id] = (i<=valIndex);
	}

	for(auto && table : tables) {
		auto begin = table.begin(), end = table.end();
		auto middle = std::stable_partition(begin,end,[&hash](RFAttribute const& entry){
			return hash[entry.id];
		});
		RAttributeTables.emplace_back(AttributeTable{middle,end});
		table.resize(std::distance(begin,middle));
		LAttributeTables.emplace_back(std::move(table));
	}
	tables.clear(); tables.shrink_to_fit();
}


///Generates a random set of indices
set<std::size_t> RFTrainer::generateRandomTableIndices(Rng::rng_type &rng) const {
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
	//For each column
	for(std::size_t j=0; j<m_inputDimension; j++){
		tables.push_back(AttributeTable{});
		auto& table = tables[j];
		table.reserve(n_elements);

		//For each row, store Attribute value, class and rowId
		for(std::size_t i=0; i<n_elements; i++){
			table.push_back(RFAttribute{elements[i].input[j], i});
		}
		std::sort(table.begin(), table.end());
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

