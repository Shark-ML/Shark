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

#include <shark/Algorithms/Trainers/RFTrainer.h>
#include <shark/Models/Trees/RFClassifier.h>
#include <boost/range/algorithm_ext/iota.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
#include <shark/Data/DataView.h>
#include <set>
#include <shark/Core/OpenMP.h>

using namespace shark;
using namespace std;


//Constructor
RFTrainer::RFTrainer(){
	m_try = 0;
	m_B = 0;
	m_nodeSize = 0;
	m_OOBratio = 0;
	m_regressionLearner = false;
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

void RFTrainer::train(RFClassifier& model, const RegressionDataset& dataset)
{
	model.clearModels();   // added by TG 23.02.2015

	//TODO O.K.: i am just fixing these things for now so that they are working.

	//Store the number of input dimensions
	m_inputDimension = inputDimension(dataset);

	//Store the size of the labels
	m_labelDimension = labelDimension(dataset);

	m_regressionLearner = true;
	setDefaults();
	
	
	//we need direct element access sicne we need to generate elementwise subsets
	std::size_t subsetSize = static_cast<std::size_t>(dataset.numberOfElements()*m_OOBratio);
	DataView<RegressionDataset const> elements(dataset);


	//Generate m_B trees
	SHARK_PARALLEL_FOR(int i = 0; i < (int)m_B; ++i){
		//For each tree generate a subset of the dataset
		//generate indices of the dataset (pick k out of n elements)
		std::vector<std::size_t> subsetIndices(elements.size());
		boost::iota(subsetIndices,0);
		boost::random_shuffle(subsetIndices);
		subsetIndices.erase(subsetIndices.begin()+subsetSize,subsetIndices.end());
		//generate the dataset by copying (TODO: this is a quick fix!
		RegressionDataset dataTrain = toDataset(subset(elements,subsetIndices));
		
		std::size_t elements = dataTrain.numberOfElements();

		//RealVector cAbove;
		AttributeTables tables;
		createAttributeTables(dataTrain.inputs(), tables);

		std::vector < RealVector > labels;
		for(std::size_t i=0; i<elements; i++){
			labels.push_back(dataTrain.element(i).label);
		}

		CARTClassifier<RealVector>::SplitMatrixType splitMatrix = buildTree(tables, dataTrain, labels, 0);
		SHARK_CRITICAL_REGION{
			//model.addModel(splitMatrix);
			model.addModel(CARTClassifier<RealVector>(splitMatrix), m_inputDimension);
		}
	}
}


//Classification
void RFTrainer::train(RFClassifier& model, const ClassificationDataset& dataset)
{
	model.clearModels();

	//Store the number of input dimensions
	m_inputDimension = inputDimension(dataset);

	//Find the largest label, so we know how big the histogram should be
	m_maxLabel = numberOfClasses(dataset)-1;

	m_regressionLearner = false;
	setDefaults();

	//we need direct element access since we need to generate element-wise subsets
	std::size_t subsetSize = static_cast<std::size_t>(dataset.numberOfElements()*m_OOBratio);
	DataView<ClassificationDataset const> elements(dataset);

	//Generate m_B trees
	SHARK_PARALLEL_FOR(int i = 0; i < (int)m_B; ++i){
		//For each tree generate a subset of the dataset
		//generate indices of the dataset (pick k out of n elements)
		std::vector<std::size_t> subsetIndices(dataset.numberOfElements());
		boost::iota(subsetIndices,0);
		boost::random_shuffle(subsetIndices);
		subsetIndices.erase(subsetIndices.begin()+subsetSize,subsetIndices.end());
		//generate the dataset by copying (TODO: this is a quick fix!
		ClassificationDataset dataTrain = toDataset(subset(elements,subsetIndices));
		//Create attribute tables
		boost::unordered_map<std::size_t, std::size_t> cAbove;
		AttributeTables tables;
		createAttributeTables(dataTrain.inputs(), tables);
		createCountMatrix(dataTrain, cAbove);

		CARTClassifier<RealVector>::SplitMatrixType splitMatrix = buildTree(tables, dataTrain, cAbove, 0);
		SHARK_CRITICAL_REGION{
			//model.addModel(splitMatrix);
			model.addModel(CARTClassifier<RealVector>(splitMatrix, m_inputDimension));
		}
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



CARTClassifier<RealVector>::SplitMatrixType RFTrainer::buildTree(AttributeTables& tables, const ClassificationDataset& dataset, boost::unordered_map<std::size_t, std::size_t>& cAbove, std::size_t nodeId ){
	CARTClassifier<RealVector>::SplitMatrixType lSplitMatrix, rSplitMatrix;

	//Construct split matrix
	CARTClassifier<RealVector>::SplitInfo splitInfo;

	splitInfo.nodeId = nodeId;
	splitInfo.leftNodeId = 0;
	splitInfo.rightNodeId = 0;

	//n = Total number of cases in the dataset
	//n1 = Number of cases to the left child node
	//n2 = number of cases to the right child node
	unsigned int n, n1, n2;

	n = tables[0].size();

	bool isLeaf = false;
	if(gini(cAbove,tables[0].size())==0 || n <= m_nodeSize){
		isLeaf = true;
	}else{
		//Count matrices
		boost::unordered_map<std::size_t, std::size_t> cBelow, cBestBelow, cTmpAbove, cBestAbove;

		//Randomly select the attributes to test for split
		set<std::size_t> tableIndicies;
		generateRandomTableIndicies(tableIndicies);

		//Iterate over the chosen attributes
		set<std::size_t>::iterator it;

		//Index of attributes
		std::size_t attributeIndex, bestAttributeIndex, bestAttributeValIndex;

		//Attribute values
		double bestAttributeVal;
		double impurity, bestImpurity = n+1;

		std::size_t prev;

		for ( it=tableIndicies.begin() ; it != tableIndicies.end(); it++ ){
			attributeIndex = *it;
			cTmpAbove = cAbove;
			cBelow.clear();
			for(std::size_t i=1; i<n; i++){
				prev = i-1;

				//Update the count of the label
				cBelow[dataset.element(tables[attributeIndex][prev].id).label]++;
				cTmpAbove[dataset.element(tables[attributeIndex][prev].id).label]--;

				if(tables[attributeIndex][prev].value!=tables[attributeIndex][i].value){
					//n1 = Number of cases to the left child node
					//n2 = number of cases to the right child node
					n1 = i;
					n2 = n-n1;

					//Calculate the Gini impurity of the split
					impurity = n1*gini(cBelow,n1)+n2*gini(cTmpAbove,n2);
					if(impurity<bestImpurity){
						//Found a more pure split, store the attribute index and value
						bestImpurity = impurity;
						bestAttributeIndex = attributeIndex;
						bestAttributeValIndex = prev;
						bestAttributeVal = tables[attributeIndex][bestAttributeValIndex].value;
						cBestAbove = cTmpAbove;
						cBestBelow = cBelow;
					}
				}
			}
		}

		if(bestImpurity<n+1){
			AttributeTables rTables, lTables;
			splitAttributeTables(tables, bestAttributeIndex, bestAttributeValIndex, lTables, rTables);
			tables.clear();
			//Continue recursively

			splitInfo.attributeIndex = bestAttributeIndex;
			splitInfo.attributeValue = bestAttributeVal;
			splitInfo.leftNodeId = 2*nodeId+1;
			splitInfo.rightNodeId = 2*nodeId+2;

			lSplitMatrix = buildTree(lTables, dataset, cBestBelow, splitInfo.leftNodeId);
			rSplitMatrix = buildTree(rTables, dataset, cBestAbove, splitInfo.rightNodeId);
		}else{
			//Leaf node
			isLeaf = true;
		}

	}

	//Store entry in the splitMatrix table
	CARTClassifier<RealVector>::SplitMatrixType splitMatrix;

	if(isLeaf){
		splitInfo.label = hist(cAbove);
		splitMatrix.push_back(splitInfo);
		return splitMatrix;
	}

	splitMatrix.push_back(splitInfo);
	splitMatrix.insert(splitMatrix.end(), lSplitMatrix.begin(), lSplitMatrix.end());
	splitMatrix.insert(splitMatrix.end(), rSplitMatrix.begin(), rSplitMatrix.end());

	return splitMatrix;
}

RealVector RFTrainer::hist(boost::unordered_map<std::size_t, std::size_t> countMatrix){

	std::vector<unsigned int> histogram(m_maxLabel+1);

	unsigned int totalElements = 0;

	boost::unordered_map<std::size_t, std::size_t>::iterator it;
	for ( it=countMatrix.begin() ; it != countMatrix.end(); it++ ){
		histogram[it->first] = it->second;
		totalElements += it->second;
	}

	RealVector normHist(histogram.size());
	for(std::size_t n = 0; n < histogram.size(); n++){
		normHist[n] = double(histogram[n]) / double(totalElements);
	}

	return normHist;
}

CARTClassifier<RealVector>::SplitMatrixType RFTrainer::buildTree(AttributeTables& tables, const RegressionDataset& dataset, const std::vector<RealVector>& labels, std::size_t nodeId ){

	//Construct split matrix
	CARTClassifier<RealVector>::SplitInfo splitInfo;


	splitInfo.nodeId = nodeId;
	splitInfo.leftNodeId = 0;
	splitInfo.rightNodeId = 0;
	splitInfo.label = average(labels);

	CARTClassifier<RealVector>::SplitMatrixType splitMatrix, lSplitMatrix, rSplitMatrix;

	//n = Total number of cases in the dataset
	//n1 = Number of cases to the left child node
	//n2 = number of cases to the right child node
	std::size_t n, n1, n2;

	n = tables[0].size();
	bool isLeaf = false;
	if(n <= m_nodeSize){
		isLeaf = true;
	}else{

		//label vectors
		std::vector<RealVector> bestLabels, tmpLabels;
		RealVector labelSumAbove(m_labelDimension), labelSumBelow(m_labelDimension);

		//Randomly select the attributes to test for split
		set<std::size_t> tableIndicies;
		generateRandomTableIndicies(tableIndicies);

		//Iterate over the chosen attributes
		set<std::size_t>::iterator it;

		//Index of attributes
		std::size_t attributeIndex, bestAttributeIndex, bestAttributeValIndex;

		//Attribute values
		double bestAttributeVal;
		double impurity, bestImpurity = -1;

		std::size_t prev;
		bool doSplit = false;
		for ( it=tableIndicies.begin() ; it != tableIndicies.end(); it++ ){
			attributeIndex = *it;

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
			tables.clear();//save memory

			//Split the labels
			std::vector<RealVector> lLabels, rLabels;
			for(std::size_t i = 0; i <= bestAttributeValIndex; i++){
				lLabels.push_back(bestLabels[i]);
			}
			for(std::size_t i = bestAttributeValIndex+1; i < bestLabels.size(); i++){
				rLabels.push_back(bestLabels[i]);
			}

			//Continue recursively
			splitInfo.attributeIndex = bestAttributeIndex;
			splitInfo.attributeValue = bestAttributeVal;
			splitInfo.leftNodeId = 2*nodeId+1;
			splitInfo.rightNodeId = 2*nodeId+2;

			lSplitMatrix = buildTree(lTables, dataset, lLabels, splitInfo.leftNodeId);
			rSplitMatrix = buildTree(rTables, dataset, rLabels, splitInfo.rightNodeId);
		}else{
			//Leaf node
			isLeaf = true;
		}

	}

	if(isLeaf){
		splitMatrix.push_back(splitInfo);
		return splitMatrix;
	}

	splitMatrix.push_back(splitInfo);
	splitMatrix.insert(splitMatrix.end(), lSplitMatrix.begin(), lSplitMatrix.end());
	splitMatrix.insert(splitMatrix.end(), rSplitMatrix.begin(), rSplitMatrix.end());

	//Store entry in the splitMatrix table
	return splitMatrix;

}



/**
 * Returns the average vector of a vector of real vectors
 */
RealVector RFTrainer::average(const std::vector<RealVector>& labels){
	RealVector avg(labels[0]);
	for(std::size_t i = 1; i < labels.size(); i++){
		avg += labels[i];
	}
	return avg/labels.size();
}

double RFTrainer::totalSumOfSquares(std::vector<RealVector>& labels, std::size_t start, std::size_t length, const RealVector& sumLabel){
	if (length < 1)
		throw SHARKEXCEPTION("[RFTrainer::totalSumOfSquares] length < 1");
	if (start+length > labels.size())
		throw SHARKEXCEPTION("[RFTrainer::totalSumOfSquares] start+length > labels.size()");

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
void RFTrainer::splitAttributeTables(const AttributeTables& tables, std::size_t index, std::size_t valIndex, AttributeTables& LAttributeTables, AttributeTables& RAttributeTables){
	AttributeTable table;

	//Build a hash table for fast lookup
	boost::unordered_map<std::size_t, bool> hash;
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
void RFTrainer::generateRandomTableIndicies(set<std::size_t>& tableIndicies){
	//Draw the m_try Generate the random attributes to search for the split
	while(tableIndicies.size()<m_try){
		tableIndicies.insert( rand() % m_inputDimension);
	}
}

///Calculates the Gini impurity of a node. The impurity is defined as
///1-sum_j p(j|t)^2
///i.e the 1 minus the sum of the squared probability of observing class j in node t
double RFTrainer::gini(boost::unordered_map<std::size_t, std::size_t>& countMatrix, std::size_t n){
	double res = 0;
	boost::unordered_map<std::size_t, std::size_t>::iterator it;
	if(n){
		n = n*n;
		for ( it=countMatrix.begin() ; it != countMatrix.end(); it++ ){
			res += sqr(it->second)/(double)n;
		}
	}
	return 1-res;
}


/// Creates the attribute tables, and in the process creates a count Matrix (cAbove).
/// A dataset consisting of m input variables has m attribute tables.
/// [attribute | class/value | rid ]
void RFTrainer::createAttributeTables(Data<RealVector> const& dataset, AttributeTables& tables){
	std::size_t elements = dataset.numberOfElements();
	//Each entry in the outer vector is an attribute table
	AttributeTable table;
	RFAttribute a;
	//For each column
	for(std::size_t j=0; j<m_inputDimension; j++){
		table.clear();
		//For each row
		for(std::size_t i=0; i<elements; i++){
			//Store Attribute value, class and rid
			a.value = dataset.element(i)[j];
			a.id = i;
			table.push_back(a);
		}
		std::sort(table.begin(), table.end(), tableSort);
		//Store this attributes attribute table
		tables.push_back(table);
	}
}


void RFTrainer::createCountMatrix(const ClassificationDataset& dataset, boost::unordered_map<std::size_t, std::size_t>& cAbove){
	std::size_t elements = dataset.numberOfElements();
	for(std::size_t i = 0 ; i < elements; i++){
		cAbove[dataset.element(i).label]++;
	}
}

bool RFTrainer::tableSort(const RFAttribute& v1, const RFAttribute& v2) {
	return v1.value < v2.value;
}






