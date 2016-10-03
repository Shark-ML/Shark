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
#include <unordered_map>
#include <shark/Core/OpenMP.h>

using namespace shark;
using std::set;
using detail::cart::Index;


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

	auto seed = Rng::discrete(0,(unsigned)-1);

	auto oobPredictions = RealMatrix{n_elements,m_labelDimension};
	std::vector<std::size_t> n_predictions(n_elements);

	//Generate m_B trees
	SHARK_PARALLEL_FOR(long b = 0; b < m_B; ++b){
		Rng::rng_type rng{static_cast<unsigned>(seed + b)};
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
		auto tables = Index{trainDataView};

		auto n_trainData = tables.noRows();
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
			for(auto const i : oobIndices){
				row(oobPredictions,i) += cart(elements[i].input);
				++n_predictions[i];
			}
		}
	}

	if(m_computeOOBerror){
		for(std::size_t i=0; i<n_elements; ++i){
			row(oobPredictions,i)/=n_predictions[i];
		}
		model.computeOOBerror(oobPredictions,elements);
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

	auto seed = Rng::discrete(0,(unsigned)-1);

	UIntMatrix oobClassTally = UIntMatrix(n_elements,m_labelCardinality);

	//Generate m_B trees
	SHARK_PARALLEL_FOR(long b = 0; b < m_B; ++b){
		Rng::rng_type rng{static_cast<unsigned>(seed + b)};
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
		auto tables = Index{trainDataView};
		auto&& cFull = detail::cart::createCountVector(trainDataView,m_labelCardinality);

		TreeType tree = buildTree(tables, trainDataView, cFull, 0, rng);
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
				auto j = blas::arg_max(histogram);
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
buildTree(detail::cart::sink<Index&> tables,
		  DataView<ClassificationDataset const> const& elements,
		  ClassVector& cFull, std::size_t nodeId,
		  Rng::rng_type& rng){

	//Construct tree
	TreeType lTree, rTree, tree;
	tree.push_back(NodeInfo{nodeId});
	NodeInfo& nodeInfo = tree[0];

	//n = Total number of cases in the dataset
	std::size_t n = tables.noRows();

	if(detail::cart::gini(cFull,n)==0.0 || n <= m_nodeSize) {
		nodeInfo.label = detail::cart::hist(cFull);
		return tree;
	}

	//Randomly select the attributes to test for split
	auto tableIndices = generateRandomTableIndices(rng);

	auto split = findSplit(tables,elements,cFull,tableIndices);

	// if the purity hasn't improved, this is a leaf.
	if(!split) {
		// this shouldn't really happen: leaves ought to be cought by previous check
		// TODO(jwrigley): Why is hist only applied to leaves, when average is applied to all nodes
		nodeInfo.label = detail::cart::hist(cFull);
		return tree;
	}
	nodeInfo <<= split;
	auto lrTables = tables.split(split.splitAttribute, split.splitRow);
	//Continue recursively

	nodeInfo.leftNodeId  = (nodeId<<1)+1;   nodeInfo.rightNodeId = (nodeId<<1)+2;

	lTree = buildTree(lrTables.first, elements, split.cAbove, nodeInfo.leftNodeId, rng);
	rTree = buildTree(lrTables.second, elements, split.cBelow, nodeInfo.rightNodeId, rng);

	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(),lTree.end(),std::back_inserter(tree));
	std::move(rTree.begin(),rTree.end(),std::back_inserter(tree));
	return tree;
}

RFTrainer::Split RFTrainer::findSplit(
		Index const& tables,
		DataView<ClassificationDataset const> const& elements,
		ClassVector const& cFull,
		set<size_t> const& tableIndices) const
{
	auto n = tables.noRows();
	Split best;
	auto const cEmpty = ClassVector(m_labelCardinality);
	for (std::size_t attributeIndex : tableIndices){
		auto const& attributeTable = tables[attributeIndex];
		auto cAbove = cEmpty, cBelow = cFull;
		for(std::size_t prev=0,i=1; i<n; prev=i++){
			auto const& label = elements[attributeTable[prev].id].label;

			// Pass the label
			++cAbove[label];    --cBelow[label];
			if(attributeTable[prev].value == attributeTable[i].value) continue;

			// n1/n2 = Number of cases to the left/right of child node
			std::size_t n1 = i,    n2 = n-i;

			//Calculate the Gini impurity of the split
			double impurity = n1*detail::cart::gini(cAbove,n1)+
							  n2*detail::cart::gini(cBelow,n2);
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

TreeType RFTrainer::
buildTree(detail::cart::sink<Index&> tables,
		  DataView<RegressionDataset const> const& elements,
		  LabelType const& sumFull,
		  std::size_t nodeId, Rng::rng_type& rng){

	//Construct tree
	TreeType tree;
	auto n = tables.noRows();
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
	if(!split) return tree;
	nodeInfo <<= split;

	//Split the attribute tables
	auto lrTables = tables.split(split.splitAttribute, split.splitRow);

	//Continue recursively
	nodeInfo.leftNodeId = (nodeId<<1)+1;
	nodeInfo.rightNodeId = (nodeId<<1)+2;

	TreeType&& lTree = buildTree(lrTables.first, elements, split.sumAbove, nodeInfo.leftNodeId, rng);
	TreeType&& rTree = buildTree(lrTables.second, elements, split.sumBelow, nodeInfo.rightNodeId, rng);

	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(),lTree.end(),std::back_inserter(tree));
	std::move(rTree.begin(),rTree.end(),std::back_inserter(tree));
	return tree;
}

RFTrainer::Split RFTrainer::findSplit (
		Index const& tables,
		DataView<RegressionDataset const> const &elements,
		RealVector const& sumFull,
		set<size_t> const &tableIndices) const
{
	auto n = tables.noRows();
	Split best{};
	LabelType const sumEmpty(m_labelDimension,0);
	for (std::size_t const attributeIndex : tableIndices){
		auto const& attributeTable = tables[attributeIndex];
		auto sumBelow = sumFull, sumAbove = sumEmpty;

		for(std::size_t prev=0,i=1; i<n; prev=i++){
			auto const& label = elements[attributeTable[prev].id].label;
			sumAbove += label; sumBelow -= label;
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


