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
#include <boost/range/algorithm/copy.hpp>

using namespace shark;
using namespace std;

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
		std::size_t numTrainElements = dataTrain.numberOfElements();

		AttributeTables tables = createAttributeTables(dataTrain.inputs());

		std::vector < RealVector > labels(numTrainElements);
		boost::copy(dataTrain.labels().elements(),labels.begin());
		//Build tree form this fold
		CARTClassifier<RealVector>::TreeType tree = buildTree(tables, dataTrain, labels, 0, dataTrain.numberOfElements());
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
        SHARK_CHECK(bestTree.size() > 0, "We should never set a tree that is empty.");
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
		//O.K. TODO: std::vector<unsigned int> is sufficient
		auto cAbove = createCountVector(dataTrain);
		AttributeTables tables = createAttributeTables(dataTrain.inputs());
		

		//create initial tree for the fold
		CARTClassifier<RealVector>::TreeType tree = buildTree(tables, dataTrain, cAbove, 0);
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
        SHARK_CHECK(bestTree.size() > 0, "We should never set a tree that is empty.");
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
	//while(i<tree.size() && tree[i].nodeId!=nodeId){
	while(tree[i].nodeId != nodeId){
		i++;
	}
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
buildTree(AttributeTables const &tables,
		  ClassificationDataset const &dataset,
		  ClassVector &cAbove, std::size_t nodeId) {
    //Construct tree, and
	// calculate the label of the node, which is the propability of class c
	// given all points in this split for every class
	ModelType::NodeInfo nodeInfo{nodeId,hist(cAbove)};

	// calculate the misclassification propability,
	// 1-p(j*|t) where j* is the class the node t is most likely to belong to;
	nodeInfo.misclassProp = 1- *std::max_element(nodeInfo.label.begin(), nodeInfo.label.end());

	//calculate leaves from the data

	//n = Total number of cases in the split
	std::size_t n = tables[0].size();

	if(!(gini(cAbove,n)==0 || n <= m_nodeSize)){
		//Count matrices


		//search the split with the best impurity
		double bestImpurity = n+1.0;
		std::size_t bestAttributeIndex = 0, bestAttributeValIndex = 0;//index of the best split
		ClassVector cBelow(m_labelCardinality),cBestBelow(m_labelCardinality),
				cBestAbove(m_labelCardinality);//labels of best split

		for (std::size_t attributeIndex=0; attributeIndex < m_inputDimension; ++attributeIndex){
			AttributeTable const& table = tables[attributeIndex];
			auto cTmpAbove = cAbove;
			std::fill(cBelow.begin(),cBelow.end(),0);
			for(std::size_t i=0; i<n-1; ++i){//go through all possible splits
				//Update the count classes of both splits after element i moved to the left split
				unsigned int label = dataset.element(table[i].id).label;
				++cBelow[label];
				--cTmpAbove[label];

				if(table[i].value != table[i+1].value){
					//n1 = Number of cases to the left child node
					//n2 = number of cases to the right child node
					std::size_t n1 = i+1;
					std::size_t n2 = n-n1;

					//Calculate the Gini impurity of the split
					double impurity = n1*gini(cBelow,n1)+n2*gini(cTmpAbove,n2);
					if(impurity < bestImpurity){
						//Found a more pure split, store the attribute index and value
						bestImpurity = impurity;
						bestAttributeIndex = attributeIndex;
						bestAttributeValIndex = i;
						cBestAbove = cTmpAbove;
						cBestBelow = cBelow;
					}
				}
			}
		}

		//std::cout<<"impurity"<<bestImpurity<<" "<<n+1<<std::endl;
		if(bestImpurity<n+1){
			double bestAttributeVal = tables[bestAttributeIndex][bestAttributeValIndex].value;
			AttributeTables rTables, lTables;
			splitAttributeTables(tables, bestAttributeIndex, bestAttributeValIndex, lTables, rTables);
			//Continue recursively
			nodeInfo.attributeIndex = bestAttributeIndex;
			nodeInfo.attributeValue = bestAttributeVal;


			//Store entry in the tree
			nodeInfo.leftNodeId = nodeId+1;
			TreeType lTree = buildTree(lTables, dataset, cBestBelow, nodeInfo.leftNodeId);
			nodeInfo.rightNodeId = nodeInfo.leftNodeId+ lTree.size();
			TreeType rTree = buildTree(rTables, dataset, cBestAbove, nodeInfo.rightNodeId);

			TreeType tree;
			tree.push_back(nodeInfo);
			tree.insert(tree.end(), lTree.begin(), lTree.end());
			tree.insert(tree.end(), rTree.begin(), rTree.end());
			return tree;
		}
	}

	TreeType tree;
	tree.push_back(nodeInfo);
	return tree;
}

RealVector CARTTrainer::hist(ClassVector const& countVector) const {

	//create a normed histogram
	std::size_t totalElements = 0;
	RealVector normedHistogram(m_labelCardinality);
	normedHistogram.clear();
	for (std::size_t i=0, s=countVector.size(); i<s; ++i){
		normedHistogram(i) = countVector[i];
		totalElements += countVector[i];
	}
	normedHistogram /= totalElements;
	return normedHistogram;
}



//Build CART tree in the regression case
CARTTrainer::TreeType CARTTrainer::
buildTree(AttributeTables const& tables,
		  RegressionDataset const& dataset,
		  std::vector<RealVector> const& labels,
		  std::size_t nodeId, std::size_t trainSize){

	//Construct tree
	ModelType::NodeInfo nodeInfo{nodeId,mean(labels)};

	//Store the Total Sum of Squares (TSS)
	RealVector labelSum = labels[0];
	for(std::size_t i=1; i< labels.size(); i++){
		labelSum += labels[0];
	}

	nodeInfo.misclassProp = totalSumOfSquares(labels, 0, labels.size(), labelSum)*((double)dataset.numberOfElements()/trainSize);

	TreeType tree, lTree, rTree;

	//n = Total number of cases in the dataset
	//n1 = Number of cases to the left child node
	//n2 = number of cases to the right child node
	std::size_t n, n1, n2;

	n = tables[0].size();

	if(n > m_nodeSize){
		//label vectors
		std::vector<RealVector> bestLabels, tmpLabels;
		RealVector labelSumAbove(m_labelDimension), labelSumBelow(m_labelDimension);

		//Index of attributes
		std::size_t bestAttributeIndex = 0, bestAttributeValIndex = 0;

		//Attribute values
		double bestAttributeVal;
		double impurity, bestImpurity = -1;

		std::size_t prev;
		bool doSplit = false;
		for ( std::size_t attributeIndex = 0; attributeIndex< m_inputDimension; attributeIndex++){

			labelSumBelow.clear();
			labelSumAbove.clear();

			tmpLabels.clear();
			//Create a labels table, that corresponds to the sorted attribute
			for(std::size_t k=0; k<tables[attributeIndex].size(); k++){
				tmpLabels.push_back(dataset.element(tables[attributeIndex][k].id).label);
				labelSumBelow += dataset.element(tables[attributeIndex][k].id).label;
			}
			labelSumAbove += tmpLabels[0];
			labelSumBelow -= tmpLabels[0];

			for(std::size_t i=1; i<n; i++){
				prev = i-1;
				if(tables[attributeIndex][prev].value!=tables[attributeIndex][i].value){
					n1=i;
					n2 = n-n1;
					//Calculate the squared error of the split
					impurity = (n1*totalSumOfSquares(tmpLabels,0,n1,labelSumAbove)+n2*totalSumOfSquares(tmpLabels,n1,n2,labelSumBelow))/(double)(n);

					if(impurity<bestImpurity || bestImpurity<0){
						//Found a more pure split, store the attribute index and value
						doSplit = true;
						bestImpurity = impurity;
						bestAttributeIndex = attributeIndex;
						bestAttributeValIndex = prev;
						bestAttributeVal = tables[attributeIndex][bestAttributeValIndex].value;
						bestLabels = tmpLabels;
					}
				}

				labelSumAbove += tmpLabels[i];
				labelSumBelow -= tmpLabels[i];
			}
		}

		if(doSplit){

			//Split the attribute tables
			AttributeTables rTables, lTables;
			splitAttributeTables(tables, bestAttributeIndex, bestAttributeValIndex, lTables, rTables);

			//Split the labels
			std::vector<RealVector> lLabels, rLabels;
			for(std::size_t i = 0; i <= bestAttributeValIndex; i++){
				lLabels.push_back(bestLabels[i]);
			}
			for(std::size_t i = bestAttributeValIndex+1; i < bestLabels.size(); i++){
				rLabels.push_back(bestLabels[i]);
			}

			//Continue recursively
			nodeInfo.attributeIndex = bestAttributeIndex;
			nodeInfo.attributeValue = bestAttributeVal;
			nodeInfo.leftNodeId = nodeId+1;
			lTree = buildTree(lTables, dataset, lLabels, nodeInfo.leftNodeId, trainSize);
                        nodeInfo.rightNodeId = nodeInfo.leftNodeId + lTree.size();
                        rTree = buildTree(rTables, dataset, rLabels, nodeInfo.rightNodeId, trainSize);
		}
	}


	tree.push_back(nodeInfo);
	tree.insert(tree.end(), lTree.begin(), lTree.end());
	tree.insert(tree.end(), rTree.begin(), rTree.end());

	//Store entry in the tree
	return tree;

}




/**
 * Returns the mean vector of a vector of real vectors
 */
RealVector CARTTrainer::mean(std::vector<RealVector> const& labels){
	RealVector avg(labels[0]);
	for(std::size_t i = 1; i < labels.size(); i++){
		avg += labels[i];
	}
	return avg/labels.size();
}

/**
 * Returns the Total Sum of Squares
 */
double CARTTrainer::totalSumOfSquares(std::vector<RealVector> const& labels, std::size_t start, std::size_t length, RealVector const& sumLabel){
	if (length < 1)
		throw SHARKEXCEPTION("[CARTTrainer::totalSumOfSquares] length < 1");
	if (start+length > labels.size())
		throw SHARKEXCEPTION("[CARTTrainer::totalSumOfSquares] start+length > labels.size()");

	RealVector labelAvg(sumLabel);
	labelAvg /= length;

	double sumOfSquares = 0;

	for(std::size_t i = 0; i < length; i++){
		sumOfSquares += norm_sqr(labels[start+i]-labelAvg);
	}
	return sumOfSquares;
}

/**
 * Returns two attribute tables: LAttrbuteTables and RAttrbuteTables
 * Calculated from splitting tables at (index, valIndex)
 */
void CARTTrainer::splitAttributeTables(AttributeTables const& tables, std::size_t index, std::size_t valIndex, AttributeTables& LAttributeTables, AttributeTables& RAttributeTables){
	AttributeTable table;

	//Build a hash table for fast lookup
	boost::unordered_map<std::size_t, bool> hash;
	for(std::size_t i = 0; i< tables[index].size(); i++){
		hash[tables[index][i].id] = i<=valIndex;
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

///Calculates the Gini impurity of a node. The impurity is defined as
///1-sum_j p(j|t)^2
///i.e the 1 minus the sum of the squared probability of observing class j in node t
double CARTTrainer::gini(ClassVector const& countVector, std::size_t n) const {
	double res = 0;
	for ( auto i : countVector){
		res += sqr(i/ static_cast<double>(n));
	}
	return 1-res;
}

/**
 * Creates the attribute tables.
 * A dataset consisting of m input variables has m attribute tables.
 * [attribute | class/value | rid ]
 */
CARTTrainer::AttributeTables CARTTrainer::createAttributeTables(Data<RealVector> const& dataset){
	std::size_t numElements = dataset.numberOfElements();
	std::size_t inputDimension = dataDimension(dataset);
	//for each input dimension an attribute table is created and stored in tables
	AttributeTables tables(inputDimension, AttributeTable(numElements));
	//For each column
	for(std::size_t j=0; j<inputDimension; j++){
		//For each row
		for(std::size_t i=0; i<numElements; i++){
			//Store Attribute value, class and element id
			tables[j][i].value = dataset.element(i)[j];
			tables[j][i].id = i;
		}
		std::sort(tables[j].begin(), tables[j].end());
	}
	return tables;
}

CARTTrainer::ClassVector CARTTrainer::
createCountVector(ClassificationDataset const& dataset) const {
	auto cAbove = ClassVector(m_labelCardinality);
	for(std::size_t i = 0 ; i < dataset.numberOfElements(); i++){
		cAbove[dataset.element(i).label]++;
	}
	return cAbove;
}





