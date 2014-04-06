
#ifndef SHARK_DATA_CVDATASETTOOLS_INL
#define SHARK_DATA_CVDATASETTOOLS_INL


#include <shark/Rng/DiscreteUniform.h>
#include <algorithm>

#include <shark/Data/DataView.h>
namespace shark{

//template<class DatasetType>
//CVFolds<DatasetType>::CVFolds(
//	DatasetType const& set, 
//	std::vector<std::size_t> const& partitionStart
//):m_dataset(set){
//	std::size_t numberOfPartitions = partitionStart.size();
//	for (std::size_t partition = 0; partition != numberOfPartitions; partition++){
//		std::size_t partitionSize=(partition+1==numberOfPartitions)?set.size():partitionStart[partition+1];
//		partitionSize-=partitionStart[partition];
//		//create the set with the indizes of the validation set of the current partition
//		IndexSet validationIndizes(partitionSize);
//		for(std::size_t batch = 0; batch != partitionSize; ++batch){
//			validationIndizes[batch]=batch+partitionStart[partition];
//		}
//		//we need the training part of the set for creation of subsets, but this is
//		//just the complement of the validation set.
//		IndexSet partitionIndizes;
//		detail::complement(validationIndizes,set.size(),partitionIndizes);
//		
//		//now add the partition to the folds.
//		m_folds.push_back(partitionIndizes);
//	}
//}
//namespace detail{
//
/////\brief Version of createCVSameSizeBalanced which works regardless of the tpe of label
/////
/////This functions needs for every class a vector which stores the indizes of the vectors  of "set" which are part of this class.
//template<class I, class L>
//CVFolds<LabeledData<I,L> > createCVSameSizeBalanced(
//	LabeledData<I,L>& set,
//	std::size_t numberOfPartitions,
//	std::vector< std::vector<std::size_t> > members,
//	std::size_t batchSize
//){
//	std::size_t numInputs = set.numberOfElements(); 
//	std::size_t numClasses = members.size();	
//
//	//shuffle elements in members
//	DiscreteUniform< Rng::rng_type > uni( shark::Rng::globalRng) ;
//	for(std::size_t c = 0; c != numClasses; c++) {
//		std::random_shuffle(members[c].begin(),members[c].end(),uni);
//	}
//
//	//calculate number of elements per validation subset as well as its starting index in
//	//the new to construct container
//	std::vector<std::size_t> validationSize(numberOfPartitions);
//	std::vector<std::size_t> validationSetStart(numberOfPartitions);
//	std::size_t nn = numInputs / numberOfPartitions;
//	std::size_t leftOver = numInputs % nn;
//	for (std::size_t partition = 0,start = 0; partition != numberOfPartitions; partition++){
//		validationSetStart[partition]=start;
//		validationSize[partition] = nn;
//		if (partition < leftOver)
//			validationSize[partition]++;
//		
//		start+=validationSize[partition];
//	}
//	
//	//calculate the size ofbatches for every validation part and their total number
//	std::vector<std::size_t> partitionStart;
//	std::vector<std::size_t> batchSizes;
//	batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);
//	
//	//construct a new set with the correct batch format from the old set
//	//(a bit ineffective here, though)
//	LabeledData<I,L> newSet(set);
//	newSet.makeIndependent();
//	newSet.repartition(batchSizes);
//	DataView<LabeledData<I,L> > oldView(set);
//	DataView<LabeledData<I,L> > newView(newSet);
//	
//	//partition classes into the validation subsets of newSet
//	std::size_t fold = 0;//current fold
//	for (std::size_t c = 0; c != numClasses; c++){
//		for(std::size_t i = 0; i != members[c].size(); i++) {
//			std::size_t oldPos= members[c][i];
//			std::size_t newPos=validationSetStart[fold];
//			newView[newPos]=oldView[oldPos];
//			
//			++validationSetStart[fold];
//			fold = (fold+1) % numberOfPartitions;
//		}
//	}
//
//	//copy newSet into the old set
//	set=newSet;
//	
//	//create folds
//	return CVFolds<LabeledData<I,L> >(set,partitionStart);
//
//}
//}//namespace detail
//
//
//template<class I,class L>
//CVFolds<LabeledData<I,L> > createCVIID(LabeledData<I,L>& set,std::size_t numberOfPartitions, std::size_t batchSize){
//	std::vector<std::size_t> indizes(set.numberOfElements());
//	for (std::size_t i=0; i != set.numberOfElements(); i++)
//		indizes[i] = Rng::discrete(0, numberOfPartitions - 1);
//	return createCVIndexed(set,numberOfPartitions,indizes,batchSize);
//}

//template<class I,class L>
//CVFolds<LabeledData<I,L> > createCVSameSize(LabeledData<I,L>& set,std::size_t numberOfPartitions,std::size_t batchSize){
//	std::size_t numInputs = set.numberOfElements();
//
//	//calculate the number of validation examples for every partition
//	std::vector<std::size_t> validationSize(numberOfPartitions);
//	std::size_t inputsForValidation = numInputs / numberOfPartitions;
//	std::size_t leftOver = numInputs - inputsForValidation * numberOfPartitions;
//	for (std::size_t i = 0; i != numberOfPartitions; i++){
//		std::size_t vs=inputsForValidation+(i<leftOver);
//		validationSize[i] =vs;
//	}
//	
//	//calculate the size of batches for every validation part and their total number
//	std::vector<std::size_t> partitionStart;
//	std::vector<std::size_t> batchSizes;
//	detail::batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);
//	
//	//this is the costly part of this function
//	//construct a new set with the correct batch format from the old set
//	LabeledData<I,L> newSet(set);
//	set.makeIndependent();
//	newSet.repartition(batchSizes);
//	//shuffle the data
//	newSet.shuffle();
//	
//	//copy new set into old set
//	set=newSet;
//
//	CVFolds<LabeledData<I,L> > folds(set,partitionStart);
//	return folds;//set;
//}

//template<class I>
//CVFolds<LabeledData<I,unsigned int> > createCVSameSizeBalanced(
//	LabeledData<I,unsigned int>& set,
//	std::size_t numberOfPartitions,
//	std::size_t batchSize
//){
//	std::size_t numInputs = set.numberOfElements(); 
//	std::size_t numClasses = numberOfClasses(set);
//	
//	//find members of each class
//	std::vector< std::vector<std::size_t> > members(numClasses);
//	for (std::size_t i = 0; i != numInputs; i++) {
//		members[set(i).label].push_back(i);
//	}
//	return detail::createCVSameSizeBalanced(set,numberOfPartitions,members,batchSize);
//
//}
//template<class I>
//CVFolds<LabeledData<I,RealVector> > createCVSameSizeBalanced(LabeledData<I,RealVector>& set,std::size_t numberOfPartitions, std::size_t batchSize){
//	std::size_t numInputs = set.numberOfElements(); 
//	
//	//calculate number of classes
//	std::size_t numClasses = numberOfClasses(set);
//	
//	//find members of each class
//	std::vector< std::vector<std::size_t> > members(numClasses);
//	for (std::size_t i = 0; i != numInputs; i++) {
//		//find class represented by label
//		//we use first max_element to get an iterator to the position of the maximum index
//		//than we calculate the distance between class 0 and the position which gives the classID
//		unsigned int classID = std::distance(
//			set(i).label.begin(),
//			std::max_element(set(i).label.begin(),set(i).label.end())
//		);
//		members[classID].push_back(i);
//	}
//	return detail::createCVSameSizeBalanced(set,numberOfPartitions,members,batchSize);
//	
//}
//
//
//template<class I,class L>
//CVFolds<LabeledData<I,L> > createCVIndexed(LabeledData<I,L>& set,std::size_t numberOfPartitions, std::vector<std::size_t> indizes,std::size_t batchSize){
//	std::size_t numInputs = set.numberOfElements();
//	SIZE_CHECK(indizes.size() == numInputs);
//	SIZE_CHECK(numberOfPartitions == *std::max_element(indizes.begin(),indizes.end())+1);
//
//	//calculate the size of validation partitions
//	std::vector<std::size_t> validationSize(numberOfPartitions,0);
//	for (std::size_t input = 0; input != numInputs; input++){
//		validationSize[indizes[input]]++;
//	}
//	
//	//calculate the start of every partition
//	std::vector<std::size_t> validationStart(numberOfPartitions,0);
//	for (std::size_t i = 1; i != numberOfPartitions; i++){
//		validationStart[i]+=validationSize[i-1]+validationStart[i-1];
//	}
//	
//	//calculate the size of batches for every validation part and their total number
//	std::vector<std::size_t> partitionStart;
//	std::vector<std::size_t> batchSizes;
//	detail::batchPartitioning(validationSize,partitionStart,batchSizes,batchSize);
//	
//	//construct a new set with the correct batch format from the old set
//	LabeledData<I,L> newSet(set);
//	newSet.makeIndependent();
//	newSet.repartition(batchSizes);
//	//create training partitions
//	for (std::size_t input = 0; input != numInputs; input++){
//		std::size_t partition = indizes[input];
//		newSet(validationStart[partition])=set(input);
//		++validationStart[partition];
//	}
//	set = newSet;
//	//now we only need to create the subset itself
//	return CVFolds<LabeledData<I,L> >(set,partitionStart);
//}


}
#endif
