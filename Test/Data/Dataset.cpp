#define BOOST_TEST_MODULE Data_Dataset
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/Dataset.h>

using namespace shark;

void testSetEquality(const UnlabeledData<int>& set1, const UnlabeledData<int>& set2){
	BOOST_REQUIRE_EQUAL(set1.size(),set2.size());

	for(size_t i=0;i!=set1.size();++i) {
		IntVector vec1=set1.batch(i);
		IntVector vec2=set2.batch(i);
		BOOST_REQUIRE_EQUAL(vec1.size(),vec2.size());
		BOOST_CHECK_EQUAL_COLLECTIONS(vec1.begin(),vec1.end(),vec2.begin(),vec2.end());
	}
}

void testDatasetEquality(const LabeledData<int, int>& set1, const LabeledData<int, int>& set2){
	BOOST_REQUIRE_EQUAL(set1.size(),set2.size());
	
	testSetEquality(set1,set2);
	testSetEquality(set1.labels(),set2.labels());
}


BOOST_AUTO_TEST_CASE( Set_Test )
{
	std::cout << "testing Set...";
	std::vector<int> inputs;

	// the test results
	std::vector<std::size_t> indizes[2];

	// fill the vectors: inputs are the number [100, ..., 199]
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	for(std::size_t i = 0; i != 10; ++i){
		indizes[i%2].push_back(i);
	}
	UnlabeledData<int> set(inputs,10);
	// 1.1 test element access and thus equality of sets
	BOOST_REQUIRE_EQUAL(set.numberOfElements(), 100u);
	for (size_t i=0; i!=100; ++i) {
		BOOST_CHECK_EQUAL(inputs[i], set(i));
	}
	//also test iterator access
	BOOST_CHECK_EQUAL_COLLECTIONS(set.elemBegin(),set.elemEnd(),inputs.begin(),inputs.end());
	
	//1.2 test batch access
	BOOST_REQUIRE_EQUAL(set.size(), 10);
	for (size_t i=0; i!=10; ++i) {
		IntVector batch=set.batch(i);
		BOOST_REQUIRE_EQUAL(batch.size(),10u);
		BOOST_CHECK_EQUAL_COLLECTIONS(
			batch.begin(),batch.end(),
			inputs.begin()+i*10,
			inputs.begin()+i*10+10
		);
	}
	
	// 2. create indexed partitions
	std::cout << "indexed...";
	UnlabeledData<int> subset;
	set.indexedSubset(indizes[0], subset);
	BOOST_REQUIRE_EQUAL(subset.size(), indizes[0].size());

	for (size_t i=0; i!=subset.size(); ++i) {
		IntVector batch=subset.batch(i);
		BOOST_REQUIRE_EQUAL(batch.size(),10);
		BOOST_CHECK_EQUAL_COLLECTIONS(
			batch.begin(),batch.end(),
			inputs.begin()+20*i,
			inputs.begin()+20*i+10
		);
	}

	// 2.1 now with complement
	std::cout << "indexed complement...";
	UnlabeledData<int> subset2;
	UnlabeledData<int> complement;
	set.indexedSubset(indizes[0], subset2, complement);
	testSetEquality(subset, subset2);

	BOOST_REQUIRE_EQUAL(complement.size(), indizes[1].size());

	for (size_t i=0; i!=complement.size(); ++i) {
		IntVector batch=complement.batch(i);
		BOOST_REQUIRE_EQUAL(batch.size(),10);
		BOOST_CHECK_EQUAL_COLLECTIONS(
			batch.begin(),batch.end(),
			inputs.begin()+20*i+10,
			inputs.begin()+20*i+20
		);
	}
}
BOOST_AUTO_TEST_CASE( Set_TrainingTestSplit_Test )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	
	//split in the middle of a batch
	UnlabeledData<int> set(inputs,10);
	splitAtElement(set,55);
	UnlabeledData<int> train = trainingSet(set);
	UnlabeledData<int> test = testSet(set);
	
	BOOST_CHECK_EQUAL(set.size(),11u);
	BOOST_CHECK_EQUAL(train.size(),6u);
	BOOST_CHECK_EQUAL(test.size(),5u);
	BOOST_CHECK_EQUAL(train.numberOfElements(),55u);
	BOOST_CHECK_EQUAL(test.numberOfElements(),45u);
	
	BOOST_CHECK_EQUAL_COLLECTIONS(
		train.elemBegin(),train.elemEnd(),
		inputs.begin(),inputs.begin()+55
	);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		test.elemBegin(),test.elemEnd(),
		inputs.begin()+55,inputs.end()
	);
}
BOOST_AUTO_TEST_CASE( Set_TrainingTestSplit_UnevenBatches_Test )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	//generate a set with unevenly sized batches
	std::vector<std::size_t> batchSizes(8);
	batchSizes[0]=8;
	batchSizes[1]=24;
	batchSizes[2]=8;
	batchSizes[3]=7;
	batchSizes[4]=8;
	batchSizes[5]=12;
	batchSizes[6]=25;
	batchSizes[7]=8;
	UnlabeledData<int> settemp(inputs,10);
	UnlabeledData<int> set(settemp,batchSizes);
	
	//split in the middle of a batch
	
	splitAtElement(set,53);
	UnlabeledData<int> train = trainingSet(set);
	UnlabeledData<int> test = testSet(set);
	
	BOOST_CHECK_EQUAL(set.size(),9u);
	BOOST_CHECK_EQUAL(train.size(),5u);
	BOOST_CHECK_EQUAL(test.size(),4u);
	BOOST_CHECK_EQUAL(train.numberOfElements(),53u);
	BOOST_CHECK_EQUAL(test.numberOfElements(),47u);
	
	BOOST_CHECK_EQUAL_COLLECTIONS(
		train.elemBegin(),train.elemEnd(),
		inputs.begin(),inputs.begin()+53
	);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		test.elemBegin(),test.elemEnd(),
		inputs.begin()+53,inputs.end()
	);
}
BOOST_AUTO_TEST_CASE( Set_TrainingTestSplit_Boundary_Test )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	
	//split between batches
	UnlabeledData<int> set(inputs,10);
	splitAtElement(set,40);
	UnlabeledData<int> train = trainingSet(set);
	UnlabeledData<int> test = testSet(set);
	
	BOOST_CHECK_EQUAL(set.size(),10);
	BOOST_CHECK_EQUAL(train.size(),4u);
	BOOST_CHECK_EQUAL(test.size(),6u);
	BOOST_CHECK_EQUAL(train.numberOfElements(),40u);
	BOOST_CHECK_EQUAL(test.numberOfElements(),60u);
	
	BOOST_CHECK_EQUAL_COLLECTIONS(
		train.elemBegin(),train.elemEnd(),
		inputs.begin(),inputs.begin()+40
	);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		test.elemBegin(),test.elemEnd(),
		inputs.begin()+40,inputs.end()
	);
}
BOOST_AUTO_TEST_CASE( Set_TrainingTestSplit_UnevenBatches_Boundary_Test )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	//generate a set with unevenly sized batches
	std::vector<std::size_t> batchSizes(8);
	batchSizes[0]=8;
	batchSizes[1]=24;
	batchSizes[2]=8;
	batchSizes[3]=7;
	batchSizes[4]=8;
	batchSizes[5]=12;
	batchSizes[6]=25;
	batchSizes[7]=8;
	UnlabeledData<int> settemp(inputs,10);
	UnlabeledData<int> set(settemp,batchSizes);
	
	//split in the middle of a batch
	
	splitAtElement(set,55);
	UnlabeledData<int> train = trainingSet(set);
	UnlabeledData<int> test = testSet(set);
	
	BOOST_CHECK_EQUAL(set.size(),8u);
	BOOST_CHECK_EQUAL(train.size(),5u);
	BOOST_CHECK_EQUAL(test.size(),3u);
	BOOST_CHECK_EQUAL(train.numberOfElements(),55u);
	BOOST_CHECK_EQUAL(test.numberOfElements(),45u);
	
	BOOST_CHECK_EQUAL_COLLECTIONS(
		train.elemBegin(),train.elemEnd(),
		inputs.begin(),inputs.begin()+55
	);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		test.elemBegin(),test.elemEnd(),
		inputs.begin()+55,inputs.end()
	);
}

// 	// 2.2 range version
// 	std::cout << "range...";
// 	UnlabeledData<int> rangeSubset, rangeSubset2;
// 	UnlabeledData<int> rangeComplement;
// 	set.rangeSubset(4, rangeSubset, rangeComplement);
// 	set.rangeSubset(4, rangeSubset2);
// 	BOOST_REQUIRE_EQUAL(rangeSubset.size(), 4);
// 	BOOST_REQUIRE_EQUAL(rangeComplement.size(), 6);
// 	for(size_t i=0; i != 40; ++i) {
// 		BOOST_CHECK_EQUAL(i+100, rangeSubset(i));
// 		BOOST_CHECK_EQUAL(i+100, rangeSubset2(i));
// 	}
// 	for(size_t i=0; i != 60; ++i) {
// 		BOOST_CHECK_EQUAL(i+140, rangeComplement(i));
// 	}
// 
// 	// 2.3 random set
// 	std::cout << "random...";
// 	// just acheck for compile errors and sanity
// 	UnlabeledData<int> randomSubset;
// 	UnlabeledData<int> randomComplement;
// 	set.randomSubset(12, randomSubset, randomComplement);
// 	BOOST_REQUIRE_EQUAL(randomSubset.size(), 12);
// 	BOOST_REQUIRE_EQUAL(randomComplement.size(), 8);
// 
// 	// 2.4 subsets from subsets
// 	std::cout << "subsubset...";
// 	UnlabeledData<int> subsubset;
// 	subset.rangeSubset(5, subsubset);
// 	BOOST_REQUIRE_EQUAL(subsubset.size(), 5);
// 	for(size_t i=0; i!=subsubset.size(); ++i) {
// 		BOOST_CHECK_EQUAL(indizes[0][i] + 20, subsubset(i));
// 	}
// 
// 	// 2.5 packing sets and checking equality
// 	//2.5.1 equality
// 	std::cout<<"equal1...";
// 	UnlabeledData<int> subset3 = subset2;
// 	BOOST_CHECK_EQUAL(subset3 == subset2,true);
// 	BOOST_CHECK_EQUAL(subset3 != subset2,false);
// 	//2.5.2 pack
// 	std::cout<<"pack...";
// 	subset2.pack();
// 	testSetEquality(subset,subset2);
// 	//2.5.3 and equality again
// 	std::cout<<"equal2...";
// 	BOOST_CHECK_EQUAL(subset3 == subset2,false);
// 	BOOST_CHECK_EQUAL(subset3 != subset2,true);
// 
// 
// 	// 3. create and query subset
// 	std::cout << "subset...";
// 	set.createNamedSubset("Dataset_test", indizes[0]);
// 	BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test"), true);
// 
// 	UnlabeledData<int> namedSet;
// 	set.namedSubset("Dataset_test", namedSet);
// 	testSetEquality(namedSet, subset);
// 	BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);
// 
// 	// 3.1 now with complement
// 	std::cout << "subset complement...";
// 	UnlabeledData<int> namedComplement;
// 	set.namedSubset("Dataset_test", namedSet, namedComplement);
// 	testSetEquality(namedSet, subset);
// 	testSetEquality(namedComplement, complement);
// 	BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);
// 	BOOST_REQUIRE_EQUAL(namedComplement.hasNamedSubset("Dataset_test"), false);
// 
// 	//3.2 create from DataSet
// 	std::cout<<"SetSubset...";
// 	set.createNamedSubset("Dataset_test2",randomSubset);
// 	BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test2"),true);
// 
// 	UnlabeledData<int> namedSet2;
// 	set.namedSubset("Dataset_test2",namedSet2);
// 	testSetEquality(namedSet2,randomSubset);
// 
// 
// 	// 4. copy constructor, default constructor and equality
// 	// a bit late, but...
// 	std::cout << "copy...";
// 	UnlabeledData<int> copy(set);
// 	testSetEquality(copy, set);
// 	BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), true);
// 	UnlabeledData<int> defaultSet;
// 	BOOST_REQUIRE_EQUAL(defaultSet.size(), 0);
// 	copy = defaultSet;
// 	BOOST_REQUIRE_EQUAL(copy.size(), 0);
// 	BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), false);

	//~ // 2.4 subsets from subsets
	//~ std::cout << "subsubset...";
	//~ UnlabeledData<int> subsubset;
	//~ subset.rangeSubset(5, subsubset);
	//~ BOOST_REQUIRE_EQUAL(subsubset.size(), 5);
	//~ for(size_t i=0; i!=subsubset.size(); ++i) {
		//~ BOOST_CHECK_EQUAL(indizes[0][i] + 20, subsubset(i));
	//~ }

	//~ // 2.5 packing sets and checking equality
	//~ //2.5.1 equality
	//~ std::cout<<"equal1...";
	//~ UnlabeledData<int> subset3 = subset2;
	//~ BOOST_CHECK_EQUAL(subset3 == subset2,true);
	//~ BOOST_CHECK_EQUAL(subset3 != subset2,false);
	//~ //2.5.2 pack
	//~ std::cout<<"pack...";
	//~ subset2.pack();
	//~ testSetEquality(subset,subset2);
	//~ //2.5.3 and equality again
	//~ std::cout<<"equal2...";
	//~ BOOST_CHECK_EQUAL(subset3 == subset2,false);
	//~ BOOST_CHECK_EQUAL(subset3 != subset2,true);


	//~ // 3. create and query subset
	//~ std::cout << "subset...";
	//~ set.createNamedSubset("Dataset_test", indizes[0]);
	//~ BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test"), true);

	//~ UnlabeledData<int> namedSet;
	//~ set.namedSubset("Dataset_test", namedSet);
	//~ testSetEquality(namedSet, subset);
	//~ BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);

	//~ // 3.1 now with complement
	//~ std::cout << "subset complement...";
	//~ UnlabeledData<int> namedComplement;
	//~ set.namedSubset("Dataset_test", namedSet, namedComplement);
	//~ testSetEquality(namedSet, subset);
	//~ testSetEquality(namedComplement, complement);
	//~ BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);
	//~ BOOST_REQUIRE_EQUAL(namedComplement.hasNamedSubset("Dataset_test"), false);

	//~ //3.2 create from DataSet
	//~ std::cout<<"SetSubset...";
	//~ set.createNamedSubset("Dataset_test2",randomSubset);
	//~ BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test2"),true);

	//~ UnlabeledData<int> namedSet2;
	//~ set.namedSubset("Dataset_test2",namedSet2);
	//~ testSetEquality(namedSet2,randomSubset);
	
	//~ //3.3 get indices back
	//~ std::vector< std::size_t > ret_ind;
	//~ set.namedSubsetIndices( "Dataset_test", ret_ind );
	//~ for ( unsigned int i=0; i<10; i++ ) {
		//~ BOOST_REQUIRE_EQUAL( ret_ind[i], 2*i );
	//~ }
	
//COMMENTED OUT BECAUSE OF JENKINS BUILD SERVER PROBLEMS WITH READ/WRITE COMMANDS!
//	//4. split file; read and write
//	std::vector< std::size_t > reta, retb, ret2a, ret2b;
//	set.createSubsetFileFromSubset( "Dataset_test", "test_data/splitfile_test.split" );
//	set.createSubsetFileFromSubset( "Dataset_test2", "test_data/splitfile_test2.split" );
//	set.createSubsetFromFile( "test_data/splitfile_test.split" );
//	set.createSubsetFromFile( "test_data/splitfile_test.split", "my_new_set" );
//	set.createSubsetFromFile( "test_data/splitfile_test2.split", "my_random_set" );
////	set.createSubsetFromFile( "test_data/splitfile_test2.split", "my_random_set" ); //this should fail
//	// obtain indices that were written out and reloaded
//	set.namedSubsetIndices( "set_from_file", reta );
//	set.namedSubsetIndices( "my_new_set", retb );
//	set.namedSubsetIndices( "Dataset_test2", ret2a );
//	set.namedSubsetIndices( "my_random_set", ret2b );
//	for ( unsigned int i=0; i<10; i++ ) {
//		BOOST_REQUIRE_EQUAL( reta[i], ret_ind[i] );
//		BOOST_REQUIRE_EQUAL( retb[i], reta[i] );
//		BOOST_REQUIRE_EQUAL( ret2a[i], ret2b[i] );
//	}
//	// final write-out for manual control
//	set.createSubsetFileFromSubset( "set_from_file", "test_data/set_from_file.split" );
//	set.createSubsetFileFromSubset( "my_new_set", "test_data/my_new_set.split" );
//	set.createSubsetFileFromSubset( "my_random_set", "test_data/my_random_set.split" );

	//~ // 5. copy constructor, default constructor and equality
	//~ // a bit late, but...
	//~ std::cout << "copy...";
	//~ UnlabeledData<int> copy(set);
	//~ testSetEquality(copy, set);
	//~ BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), true);
	//~ UnlabeledData<int> defaultSet;
	//~ BOOST_REQUIRE_EQUAL(defaultSet.size(), 0);
	//~ copy = defaultSet;
	//~ BOOST_REQUIRE_EQUAL(copy.size(), 0);
	//~ BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), false);
	//~ std::cout << "Set tests done" << std::endl;
//}

/*BOOST_AUTO_TEST_CASE( Dataset_Test )
{
	std::cout<<"testing Dataset...";
	std::vector<int> inputs;
	std::vector<int> labels;
	std::vector<size_t> indizes[2];
	
	//the testresults
	std::vector<int> testLabelPartition[2];
	//fill the vectors. inputs are the number [0,19], labels go from [20...39]
	for(size_t i=0;i!=20;++i){
		inputs.push_back(i);
		labels.push_back(20+i);
		indizes[i%2].push_back(i);
		testLabelPartition[i%2].push_back(20+i);
	}
	LabeledData<int,int> set(inputs,labels);

	// 1. test element access and thus equality of sets
	BOOST_REQUIRE_EQUAL(set.size(),inputs.size());

	for(size_t i=0;i!=set.size();++i){
		BOOST_CHECK_EQUAL(inputs[i],set.input(i));
		BOOST_CHECK_EQUAL(labels[i],set.label(i));
	}
	// 1.1 test creating sets
	std::cout<<"sets...";
	UnlabeledData<int> inputSet=set.inputs();
	Data<int> labelSet=set.labels();
	for(size_t i=0;i!=set.size();++i){
		BOOST_CHECK_EQUAL(inputSet(i),set.input(i));
		BOOST_CHECK_EQUAL(labelSet(i),set.label(i));
	}
	// 1.2 test creating datasets from sets
	LabeledData<int,int> setCopySet(inputSet,labelSet);
	testDatasetEquality(setCopySet,set);

	// 2. create indexed partitions
	std::cout<<"indexed...";
	LabeledData<int,int> subset;
	set.indexedSubset(indizes[0],subset);
	BOOST_REQUIRE_EQUAL(subset.size(),indizes[0].size());

	for(size_t i=0;i!=subset.size();++i){
		BOOST_CHECK_EQUAL(indizes[0][i],subset.input(i));
		BOOST_CHECK_EQUAL(indizes[0][i]+20,subset.label(i));
	}

	// 2.1 now with complement
	std::cout<<"indexed complement...";
	LabeledData<int,int> subset2;
	LabeledData<int,int> complement;
	set.indexedSubset(indizes[0],subset2,complement);
	testDatasetEquality(subset,subset2);

	BOOST_REQUIRE_EQUAL(complement.size(),indizes[1].size());

	for(size_t i=0;i!=subset2.size();++i){
		BOOST_CHECK_EQUAL(indizes[1][i],complement.input(i));
		BOOST_CHECK_EQUAL(indizes[1][i]+20,complement.label(i));
	}


	// 2.2 range version
	std::cout<<"range...";
	LabeledData<int,int> rangeSubset,rangeSubset2;
	LabeledData<int,int> rangeComplement;
	set.rangeSubset(10,rangeSubset,rangeComplement);
	BOOST_REQUIRE_EQUAL(rangeSubset.size(),10);
	BOOST_REQUIRE_EQUAL(rangeComplement.size(),10);
	for(size_t i=0;i!=rangeSubset.size();++i){
		BOOST_CHECK_EQUAL(i+20,rangeSubset.label(i));
		BOOST_CHECK_EQUAL(i+30,rangeComplement.label(i));
	}
	set.rangeSubset(10,rangeSubset2);
	testDatasetEquality(rangeSubset,rangeSubset2);


	// 2.3 random set
	std::cout<<"random...";
	// just a check for compile errors and sanity
	LabeledData<int,int> randomSubset;
	LabeledData<int,int> randomComplement;
	set.randomSubset(12,randomSubset,randomComplement);
	BOOST_REQUIRE_EQUAL(randomSubset.size(),12);
	BOOST_REQUIRE_EQUAL(randomComplement.size(),8);

	// 2.4 subsets from subsets
	std::cout<<"subsubset...";
	LabeledData<int,int> subsubset;
	subset.rangeSubset(5,subsubset);
	BOOST_REQUIRE_EQUAL(subsubset.size(),5);
	for(size_t i=0;i!=subsubset.size();++i){
		BOOST_CHECK_EQUAL(indizes[0][i],subsubset.input(i));
		BOOST_CHECK_EQUAL(indizes[0][i]+20,subsubset.label(i));
	}

	// 2.5 packing sets and checking equality
	//2.5.1 equality
	std::cout<<"equal1...";
	LabeledData<int,int> subset3 = subset2;
	BOOST_CHECK_EQUAL(subset3 == subset2,true);
	BOOST_CHECK_EQUAL(subset3 != subset2,false);
	//2.5.2 pack
	std::cout<<"pack...";
	subset2.pack();
	testDatasetEquality(subset,subset2);
	//2.5.3 and equality again
	std::cout<<"equal2...";
	BOOST_CHECK_EQUAL(subset3 == subset2,false);
	BOOST_CHECK_EQUAL(subset3 != subset2,true);

	// 3. create and query subset
	std::cout<<"subset...";
	set.createNamedSubset("Dataset_test",indizes[0]);
	BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test"),true);

	LabeledData<int,int> namedSet;
	set.namedSubset("Dataset_test",namedSet);
	testDatasetEquality(namedSet,subset);
	BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"),false);

	// 3.1 now with complement
	std::cout<<"subset complement...";
	LabeledData<int,int> namedComplement;
	set.namedSubset("Dataset_test",namedSet,namedComplement);
	testDatasetEquality(namedSet,subset);
	testDatasetEquality(namedComplement,complement);
	BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"),false);
	BOOST_REQUIRE_EQUAL(namedComplement.hasNamedSubset("Dataset_test"),false);

	//3.2 create from DataSet
	std::cout<<"DataSubset...";
	set.createNamedSubset("Dataset_test2",randomSubset);
	BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test2"),true);

	LabeledData<int,int> namedSet2;
	set.namedSubset("Dataset_test2",namedSet2);
	testDatasetEquality(namedSet2,randomSubset);



	// 4. copy constructor, default constructor and equality
	// a bit late, but...
	std::cout<<"copy...";
	LabeledData<int,int> copy(set);
	testDatasetEquality(copy,set);
	BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"),true);
	LabeledData<int,int> defaultSet;
	BOOST_REQUIRE_EQUAL(defaultSet.size(),0);
	copy = defaultSet;
	BOOST_REQUIRE_EQUAL(copy.size(),0);
	BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"),false);

	std::cout<<"Dataset tests done"<<std::endl;
}*/
