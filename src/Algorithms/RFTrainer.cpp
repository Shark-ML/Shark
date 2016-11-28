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
using detail::cart::SortedIndex;
using detail::cart::createCountVector;
using detail::cart::hist;
using detail::cart::Bag;
using detail::cart::bootstrap;


//Constructor
RFTrainer::RFTrainer(bool computeFeatureImportances, bool computeOOBerror)
: m_B{100}, m_OOBratio{0.66}
{
	m_try = 0;
	m_nodeSize = 0;
	m_regressionLearner = false;
	m_computeFeatureImportances = computeFeatureImportances;
	m_computeOOBerror = computeOOBerror;
	m_computeCARTOOBerror = false;
	m_bootstrapWithReplacement = false;
	m_impurityMeasure = ImpurityMeasure::gini;
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
	m_impurityFn = setImpurityFn(m_impurityMeasure);
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

	m_regressionLearner = true;
	setDefaults();
	
	//we need direct element access since we need to generate elementwise subsets
	DataView<RegressionDataset const> elements(dataset);
	auto const n_elements = elements.size();
	std::size_t subsetSize = static_cast<std::size_t>(n_elements*m_OOBratio);

	auto seed = Rng::discrete(0,(unsigned)-1);

	auto oobPredictions = RealMatrix{n_elements,m_labelDimension};
	std::vector<std::size_t> n_predictions(n_elements);

	//Generate m_B trees
	SHARK_PARALLEL_FOR(long b = 0; b < m_B; ++b){
		Rng::rng_type rng{static_cast<unsigned>(seed + b)};

		auto bag = bootstrap(elements, rng, subsetSize, m_bootstrapWithReplacement);

		//Create attribute tables
		auto tables = SortedIndex{bag.dataView()};
		auto sumFull = detail::cart::sum<RealVector>(tables.noRows(), [&](std::size_t i){
			return bag.dataView()[i].label;
		});

		TreeType tree = buildTree(std::move(tables), bag.dataView(), sumFull, 0, rng);
		CARTType cart(std::move(tree), m_inputDimension);

		// if oob error or importances have to be computed, create an oob sample
		if(m_computeCARTOOBerror || m_computeFeatureImportances){
			RegressionDataset dataOOB = toDataset(bag.oobDataView());

			// cart oob errors are computed implicitly whenever importances are
			if(m_computeFeatureImportances){
				cart.computeFeatureImportances(dataOOB, rng);
			} else cart.computeOOBerror(dataOOB);
		}

		SHARK_CRITICAL_REGION{
			model.addModel(cart);
			for(auto const i : bag.oobIndices){
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

	m_regressionLearner = false;
	setDefaults();

	//we need direct element access since we need to generate element-wise subsets
	DataView<ClassificationDataset const> elements(dataset);
	auto const n_elements = elements.size();
	std::size_t subsetSize = static_cast<std::size_t>(n_elements*m_OOBratio);

	auto seed = Rng::discrete(0,(unsigned)-1);

	UIntMatrix oobClassTally(n_elements,m_labelCardinality);

	//Generate m_B trees
	SHARK_PARALLEL_FOR(long b = 0; b < m_B; ++b){
		Rng::rng_type rng{static_cast<unsigned>(seed + b)};

		auto bag = bootstrap(elements, rng, subsetSize, m_bootstrapWithReplacement);

		//Create attribute tables
		auto tables = SortedIndex{bag.dataView()};
		auto&& cFull = createCountVector(bag.dataView(),m_labelCardinality);

		TreeType tree = buildTree(std::move(tables), bag.dataView(), cFull, 0, rng);
		CARTType cart(std::move(tree), m_inputDimension);

		// if oob error or importances have to be computed, create an oob sample
		if(m_computeCARTOOBerror || m_computeFeatureImportances){
			ClassificationDataset dataOOB = toDataset(bag.oobDataView());

			// if importances should be computed, oob errors are computed implicitly
			if(m_computeFeatureImportances){
				cart.computeFeatureImportances(dataOOB, rng);
			} else cart.computeOOBerror(dataOOB);
		}

		SHARK_CRITICAL_REGION{
			model.addModel(cart);
			for(auto const i : bag.oobIndices){
				auto histogram = cart(elements[i].input);
				auto j = arg_max(histogram);
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
buildTree(SortedIndex&& tables,
		  DataView<ClassificationDataset const> const& elements,
		  ClassVector& cFull, std::size_t nodeId,
		  Rng::rng_type& rng){

	//Construct tree
	TreeType tree;
	tree.push_back(NodeInfo{nodeId});
	NodeInfo& nodeInfo = tree[0];

	//n = Total number of cases in the dataset
	std::size_t n = tables.noRows();

	if(m_impurityFn(cFull,n)==0.0 || n <= m_nodeSize) {
		nodeInfo.label = hist(cFull);
		return tree;
	}

	//Randomly select the attributes to test for split
	auto tableIndices = generateRandomTableIndices(rng);

	auto split = findSplit(tables,elements,cFull,tableIndices);

	// if the purity hasn't improved, this is a leaf.
	if(!split) {
		// this shouldn't really happen: leaves ought to be cought by previous check
		// TODO(jwrigley): Why is hist only applied to leaves, when average is applied to all nodes
		nodeInfo.label = hist(cFull);
		return tree;
	}
	nodeInfo <<= split;
	auto lrTables = tables.split(split.splitAttribute, split.splitRow);
	//Continue recursively

	nodeInfo.leftNodeId = nodeId+1;
	TreeType lTree = buildTree(std::move(lrTables.first), elements, split.cAbove, nodeInfo.leftNodeId, rng);

	nodeInfo.rightNodeId = nodeInfo.leftNodeId + lTree.size();
	TreeType rTree = buildTree(std::move(lrTables.second), elements, split.cBelow, nodeInfo.rightNodeId, rng);

	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(),lTree.end(),std::back_inserter(tree));
	std::move(rTree.begin(),rTree.end(),std::back_inserter(tree));
	return tree;
}

RFTrainer::Split RFTrainer::findSplit(
		SortedIndex const& tables,
		DataView<ClassificationDataset const> const& elements,
		ClassVector const& cFull,
		set<size_t> const& tableIndices) const
{
	auto n = tables.noRows();
	Split best;
	ClassVector cAbove(m_labelCardinality);
	for (std::size_t attributeIndex : tableIndices){
		auto const& attributeTable = tables[attributeIndex];
		auto cBelow = cFull; cAbove.clear();
		for(std::size_t prev=0,i=1; i<n; prev=i++){
			auto const& label = elements[attributeTable[prev].id].label;

			// Pass the label
			++cAbove[label];    --cBelow[label];
			if(attributeTable[prev].value == attributeTable[i].value) continue;

			// n1/n2 = Number of cases to the left/right of child node
			std::size_t n1 = i,    n2 = n-i;

			//Calculate the Gini impurity of the split
			double impurity = n1*m_impurityFn(cAbove,n1)+
							  n2*m_impurityFn(cBelow,n2);
			if(impurity<best.impurity){
				//Found a more pure split, store the attribute index and value
				best.splitAttribute = attributeIndex;
				best.splitRow = prev;
				best.impurity = impurity;
				best.cAbove = cAbove;
				best.cBelow = cBelow;
			}
		}
	}
	best.splitValue = tables[best.splitAttribute][best.splitRow].value;
	return best;
}

TreeType RFTrainer::
buildTree(SortedIndex&& tables,
		  DataView<RegressionDataset const> const& elements,
		  LabelType const& sumFull,
		  std::size_t nodeId, Rng::rng_type& rng){

	//Construct tree
	TreeType tree;
	//n = Total number of cases in the dataset
	auto n = tables.noRows();
	// TODO(jwrigley): Why is average assigned to all nodes, when hist is only applied to leaves?
	tree.push_back(NodeInfo{nodeId,sumFull/n});
	NodeInfo& nodeInfo = tree[0];

	if(n <= m_nodeSize) return tree; // Must be leaf

	//Randomly select the attributes to test for split
	auto tableIndices = generateRandomTableIndices(rng);

	auto split = findSplit(tables, elements, sumFull,tableIndices);

	// if the purity hasn't improved, this is a leaf.
	if(!split) return tree;
	nodeInfo <<= split;

	//Split the attribute tables
	auto lrTables = tables.split(split.splitAttribute, split.splitRow);

	//Continue recursively
	nodeInfo.leftNodeId = nodeId+1;
	TreeType&& lTree = buildTree(std::move(lrTables.first), elements, split.sumAbove, nodeInfo.leftNodeId, rng);

	nodeInfo.rightNodeId = nodeInfo.leftNodeId + lTree.size();
	TreeType&& rTree = buildTree(std::move(lrTables.second), elements, split.sumBelow, nodeInfo.rightNodeId, rng);

	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(),lTree.end(),std::back_inserter(tree));
	std::move(rTree.begin(),rTree.end(),std::back_inserter(tree));
	return tree;
}

RFTrainer::Split RFTrainer::findSplit (
		SortedIndex const& tables,
		DataView<RegressionDataset const> const &elements,
		RealVector const& sumFull,
		set<size_t> const &tableIndices) const
{
	auto n = tables.noRows();
	Split best{};
	LabelType sumAbove(m_labelDimension);
	for (std::size_t const attributeIndex : tableIndices){
		auto const& attributeTable = tables[attributeIndex];
		auto sumBelow = sumFull; sumAbove.clear();

		for(std::size_t prev=0,i=1; i<n; prev=i++){
			auto const& label = elements[attributeTable[prev].id].label;
			sumAbove += label; sumBelow -= label;
			if(attributeTable[prev].value == attributeTable[i].value) continue;

			std::size_t n1=i,    n2 = n-i;
			//Calculate the squared error of the split
			double purity = norm_sqr(sumAbove)/n1 + norm_sqr(sumBelow)/n2;

			if(purity>best.purity){
				//Found a more pure split, store the attribute index and value
				best.splitAttribute = attributeIndex;
				best.splitRow = prev;
				best.purity = purity;
				best.sumAbove = sumAbove;
				best.sumBelow = sumBelow;
			}
		}
	}
	auto& bestTable  = tables[best.splitAttribute];
	best.splitValue = (bestTable[best.splitRow].value + bestTable[best.splitRow+1].value)/2.0;
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


