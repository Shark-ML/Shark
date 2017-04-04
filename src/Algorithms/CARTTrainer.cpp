/*
 * CARTTrainer.c
 *
 *  Created on: Dec 1, 2011
 *	  Author: nybohansen
 */
#define SHARK_COMPILE_DLL
#include <shark/Algorithms/Trainers/CARTTrainer.h>
#include <shark/Data/CVDatasetTools.h>
#include <shark/ObjectiveFunctions/Loss/SquaredLoss.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>

using namespace shark;
using namespace std;
using detail::cart::SortedIndex;

//Train model with a regression dataset
void CARTTrainer::train(ModelType& model, RegressionDataset const& dataset)
{
	//Store the number of input dimensions
	m_inputDimension = inputDimension(dataset);

	//Pass input dimension (i.e., number of attributes) to tree model
	model.setInputDimension(m_inputDimension);

	//Store the size of the labels
	m_labelDimension = labelDimension(dataset);

	// create cross-validation folds
	RegressionDataset set=dataset;
	CVFolds<RegressionDataset > folds = createCVSameSize(set, m_numberOfFolds);
	double bestErrorRate = std::numeric_limits<double>::max();
	CARTClassifier<RealVector>::TreeType bestTree;

	for (unsigned fold = 0; fold < m_numberOfFolds; ++fold){
		//Run through all the cross validation sets
		RegressionDataset dataTrain = folds.training(fold);
		RegressionDataset dataTest = folds.validation(fold);

		//AttributeTables tables = createAttributeTables(dataTrain.inputs());

		RealVector sumFull{0.};
		for(auto const& element: dataTrain.elements()){ sumFull += element.label; }

		//Build tree form this fold
		auto tree = buildTree(SortedIndex{dataTrain}, dataTrain, sumFull, 0, dataTrain.numberOfElements());
		//Add the tree to the model and prune
		model.setTree(tree);
		while(true){
			//evaluate the error of current tree
			SquaredLoss<> loss;
			double error = loss.eval(dataTest.labels(), model(dataTest.inputs()));

			if(error < bestErrorRate){
				//We have found a subtree that has a smaller error rate when tested!
				bestErrorRate = error;
				bestTree = tree;
			}
                        if(tree.size() == 1) break;
			pruneTree(tree);
			model.setTree(tree);
		}
	}
        SHARK_RUNTIME_CHECK(bestTree.size() > 0, "We should never set a tree that is empty.");
	model.setTree(bestTree);
}


//Classification
void CARTTrainer::train(ModelType& model, ClassificationDataset const& dataset){
	//Store the number of input dimensions
	m_inputDimension = inputDimension(dataset);

	//Pass input dimension (i.e., number of attributes) to tree model
	model.setInputDimension(m_inputDimension);

	m_labelCardinality = numberOfClasses(dataset);

	// create cross-validation folds
	ClassificationDataset set=dataset;
	CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(set, m_numberOfFolds);
	//find the best tree for the cv folds
	double bestErrorRate = std::numeric_limits<double>::max();
	CARTClassifier<RealVector>::TreeType bestTree;

	//Run through all the cross validation sets
	for (unsigned fold = 0; fold < m_numberOfFolds; ++fold) {
		ClassificationDataset dataTrain = folds.training(fold);
		ClassificationDataset dataTest = folds.validation(fold);
		//Create attribute tables
		//O.K. stores how often label(i) can be found in the dataset
		auto&& cFull = detail::cart::createCountVector(dataTrain,m_labelCardinality);


		//create initial tree for the fold
		TreeType tree = buildTree(SortedIndex{dataTrain}, dataTrain, cFull, 0);
		model.setTree(tree);

		while(true){
			ZeroOneLoss<unsigned int, RealVector> loss;
			double errorRate = loss.eval(dataTest.labels(), model(dataTest.inputs()));
			if(errorRate < bestErrorRate){
				//We have found a subtree that has a smaller error rate when tested!
				bestErrorRate = errorRate;
				bestTree = tree;
			}
                        if(tree.size()!=1) break;
			pruneTree(tree);
			model.setTree(tree);
		}
	}
        SHARK_RUNTIME_CHECK(bestTree.size() > 0, "We should never set a tree that is empty.");
	model.setTree(bestTree);

}

//TODO
//~ // train using weights
//~ void CARTTrainer::train(ModelType& model, ClassificationDataset const& dataset, RealVector const& weights, double& error) {
	//~ //Store the number of input dimensions
	//~ m_inputDimension = inputDimension(dataset);

	//~ //Find the largest label, so we know how big the histogram should be
	//~ m_labelCardinality = numberOfClasses(dataset);

	//~ // create cross-validation folds
	//~ ClassificationDataset set=dataset;
	//~ CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(set, m_numberOfFolds);
	//~ //find the best tree for the cv folds
	//~ double bestErrorRate = std::numeric_limits<double>::max();
	//~ CARTClassifier<RealVector>::TreeType bestTree;

	//~ //Run through all the cross validation sets
	//~ for (unsigned fold = 0; fold < m_numberOfFolds; ++fold) {
		//~ ClassificationDataset dataTrain = folds.training(fold);
		//~ ClassificationDataset dataTest = folds.validation(fold);
		//~ //Create attribute tables
		//~ auto cAbove = createCountVector(dataTrain);
		//~ AttributeTables tables = createAttributeTables(dataTrain.inputs());

		//~ //create initial tree for the fold
		//~ CARTClassifier<RealVector>::TreeType tree = buildTree(tables, dataTrain, cAbove, 0);
		//~ model.setTree(tree);

		//~ while(tree.size()!=1){
			//~ double errorRate = evalWeightedError(model, dataTest, weights);
			//~ if(errorRate < bestErrorRate){
				//~ //We have found a subtree that has a smaller error rate when tested!
				//~ bestErrorRate = errorRate;
				//~ bestTree = tree;
			//~ }
			//~ pruneTree(tree);
			//~ model.setTree(tree);
		//~ }
	//~ }

	//~ error = bestErrorRate;
	//~ model.setTree(bestTree);
//~ }


void CARTTrainer::pruneTree(TreeType & tree){

	//Calculate g of all the nodes
	measureStrength(tree, 0, 0);

	//Find the lowest g of the internal nodes
	double g = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != tree.size(); i++){
		if(tree[i].leftNodeId > 0 && tree[i].g < g){
			//Update g
			g = tree[i].g;
		}
	}
	//Prune the nodes with lowest g and make them terminal
	for(std::size_t i=0; i != tree.size(); i++){
		//Make the internal nodes with the smallest g terminal nodes and prune their children!
		if( tree[i].leftNodeId > 0 && tree[i].g == g){
			pruneNode(tree, tree[i].leftNodeId);
			pruneNode(tree, tree[i].rightNodeId);
			// //Make the node terminal
			tree[i].leftNodeId = 0;
			tree[i].rightNodeId = 0;
		}
	}
}

std::size_t CARTTrainer::findNode(TreeType & tree, std::size_t nodeId){
	std::size_t i = 0;
	for(auto const s = tree.size(); tree[i].nodeId != nodeId && i<s; i++);
	return i;
}

/*
	Removes branch with root node id nodeId, incl. the node itself
*/
void CARTTrainer::pruneNode(TreeType & tree, std::size_t nodeId){
	std::size_t i = findNode(tree,nodeId);

	if(tree[i].leftNodeId>0){
		//Prune left branch
		pruneNode(tree, tree[i].leftNodeId);
		//Prune right branch
		pruneNode(tree, tree[i].rightNodeId);
	}
	//Remove node
	tree.erase(tree.begin()+i);
}


void CARTTrainer::measureStrength(TreeType & tree, std::size_t nodeId, std::size_t parentNode){
	std::size_t i = findNode(tree,nodeId);

	//Reset the entries
	tree[i].r = 0;
	tree[i].g = 0;

	if(tree[i].leftNodeId==0){
		//Leaf node
		//Update number of leafs
		tree[parentNode].r+=1;
		//update R(T) from r(t) of node. R(T) is the sum of all the leaf's r(t)
		tree[parentNode].g+= tree[i].misclassProp;
	}else{

		//Left recursion
		measureStrength(tree, tree[i].leftNodeId, i);
		//Right recursion
		measureStrength(tree, tree[i].rightNodeId, i);

		if(parentNode != i){
			tree[parentNode].r+= tree[i].r;
			tree[parentNode].g+= tree[i].g;
		}

		//Final calculation of g
		tree[i].g = (tree[i].misclassProp- tree[i].g)/(tree[i].r-1);
	}
}

//Classification case
CARTTrainer::TreeType CARTTrainer::
buildTree(SortedIndex&& tables,
          ClassificationDataset const &dataset,
          ClassVector &cFull, std::size_t nodeId) {
    //Construct tree, and
	// calculate the label of the node, which is the propability of class c
	// given all points in this split for every class
	TreeType tree;
	tree.push_back(ModelType::NodeInfo{nodeId, detail::cart::hist(cFull)});
	ModelType::NodeInfo& nodeInfo = tree[0];

	//n = Total number of cases in the split
	std::size_t n = tables.noRows();

	// calculate the misclassification propability,
	// 1-p(j*|t) where j* is the class the node t is most likely to belong to;
	nodeInfo.misclassProp = 1- *std::max_element(nodeInfo.label.begin(), nodeInfo.label.end());

	//calculate leaves from the data

	if(detail::cart::gini(cFull,n)==0.0 || n <= m_nodeSize) return tree;
	//Count matrices

	//search the split with the best impurity
	auto split = findSplit(tables,dataset, cFull);

	if(!split) return tree;

	nodeInfo <<= split;
	auto lrTables = tables.split(split.splitAttribute,split.splitRow);
	//Continue recursively

	//Store entry in the tree
	nodeInfo.leftNodeId = nodeId+1;
	TreeType lTree = buildTree(std::move(lrTables.first), dataset, split.cAbove, nodeInfo.leftNodeId);

	nodeInfo.rightNodeId = nodeInfo.leftNodeId+ lTree.size();
	TreeType rTree = buildTree(std::move(lrTables.second), dataset, split.cBelow, nodeInfo.rightNodeId);

	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(), lTree.end(), std::back_inserter(tree));
	std::move(rTree.begin(), rTree.end(), std::back_inserter(tree));
	return tree;
}

CARTTrainer::Split CARTTrainer::findSplit(
		SortedIndex const& tables,
        ClassificationDataset const& dataset,
        ClassVector const& cFull) const
{
	auto const n = tables.noRows();
	Split best;
	ClassVector cAbove(m_labelCardinality);
	for (std::size_t attributeIndex=0; attributeIndex < m_inputDimension; ++attributeIndex){
		auto const& table = tables[attributeIndex];
		auto cBelow = cFull; cAbove.clear();
		for(std::size_t i=0,next=1; next<n; i=next++){//go through all possible splits
			//Update the count classes of both splits after element i moved to the left split
			unsigned int label = dataset.element(table[i].id).label;

			// Pass the label
			++cAbove[label];    --cBelow[label];
			if(table[i].value == table[next].value) continue;

			// n1/n2 = Number of cases to the left/right of child node
			std::size_t n1 = next,   n2 = n-next;

			//Calculate the Gini impurity of the split
			double impurity = n1*detail::cart::gini(cAbove,n1)+
			                  n2*detail::cart::gini(cBelow,n2);
			if(impurity < best.impurity){
				//Found a more pure split, store the attribute index and value
				best.splitAttribute = attributeIndex;
				best.splitRow = i;
				best.impurity = impurity;
				best.cAbove = cAbove;
				best.cBelow = cBelow;
			}
		}
	}
	best.splitValue = tables[best.splitAttribute][best.splitRow].value;
	return best;
}

//Build CART tree in the regression case
CARTTrainer::TreeType CARTTrainer::
buildTree(SortedIndex&& tables,
          RegressionDataset const& dataset,
          RealVector const& sumFull,
          std::size_t nodeId, std::size_t trainSize){

	//Construct tree
	TreeType tree;
	//n = Total number of cases in the dataset
	std::size_t n = tables.noRows();
	tree.push_back(ModelType::NodeInfo{nodeId,sumFull/n});
	ModelType::NodeInfo& nodeInfo = tree[0];


	// TODO(jwrigley): use alternative to nodeInfo.misclassprop so we can remove these lines {
	//Store the Total Sum of Squares (TSS)
	std::vector<RealVector> labels(dataset.numberOfElements());
	for(std::size_t i = 0; i<n; i++){
		labels[i] = dataset.element(tables[0][i].id).label;
	}
	//? trainSize seems superfluous seeing as trainSize is set to dataset.numberOfElements(), and is never changed
	nodeInfo.misclassProp = totalSumOfSquares(labels, 0, tables.noRows(), sumFull)*((double)dataset.numberOfElements()/trainSize);
	// END TODO(jwrigley): }

	if(n <= m_nodeSize) return tree;

	auto split = findSplit(tables, dataset, sumFull);

	// if the purity hasn't improved, this is a leaf.
	if(!split) return tree;
	nodeInfo <<= split;

	//Split the attribute tables
	auto lrTables = tables.split(split.splitAttribute,split.splitRow);

	//Continue recursively
	nodeInfo.leftNodeId = nodeId+1;
	TreeType lTree = buildTree(std::move(lrTables.first), dataset, split.sumAbove, nodeInfo.leftNodeId, trainSize);

	nodeInfo.rightNodeId = nodeInfo.leftNodeId + lTree.size();
	TreeType rTree = buildTree(std::move(lrTables.second), dataset, split.sumBelow, nodeInfo.rightNodeId, trainSize);


	tree.reserve(tree.size()+lTree.size()+rTree.size());
	std::move(lTree.begin(), lTree.end(), std::back_inserter(tree));
	std::move(rTree.begin(), rTree.end(), std::back_inserter(tree));

	return tree;
}

CARTTrainer::Split CARTTrainer::findSplit(
		SortedIndex const& tables,
        RegressionDataset const& dataset,
        RealVector const& sumFull) const
{
	auto n = tables.noRows();
	Split best{};
	RealVector sumAbove(labelDimension(dataset));
	for(std::size_t attributeIndex = 0; attributeIndex< m_inputDimension; attributeIndex++){
		auto const& table = tables[attributeIndex];
		auto sumBelow = sumFull; sumAbove.clear();

		for(std::size_t i=0,next=1; next<n; i=next++){
			auto const& label = dataset.element(table[i].id).label;
			sumAbove += label; sumBelow -= label;
			if(table[i].value == table[next].value) continue;

			// n1/n2 = number of cases to the left/right child node
			auto n1=next,     n2 = n-next;
			//Calculate the squared error of the split
			double purity = norm_sqr(sumAbove)/n1 + norm_sqr(sumBelow)/n2;

			if(purity>best.purity){
				//Found a more pure split, store the attribute index and value
				best.splitAttribute = attributeIndex;
				best.splitRow = i;
				best.purity = purity;
				best.sumAbove = sumAbove;
				best.sumBelow = sumBelow;
			}
		}
	}
	best.splitValue = tables[best.splitAttribute][best.splitRow].value;
	return best;
}


/**
 * Returns the Total Sum of Squares
 */
double CARTTrainer::totalSumOfSquares(std::vector<RealVector> const& labels, std::size_t start, std::size_t length, RealVector const& sumLabel){
	SIZE_CHECK(length > 1);
	SIZE_CHECK(start+length <= labels.size());

	RealVector labelAvg(sumLabel);
	labelAvg /= length;

	double sumOfSquares = 0.;

	for(std::size_t i = start, s = length+start; i < s; i++){
		sumOfSquares += distanceSqr(labels[i],labelAvg);
	}
	return sumOfSquares;
}
