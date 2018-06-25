#define BOOST_TEST_MODULE Data_Dataset
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/Dataset.h>

#include <sstream>

using namespace shark;

void testSetEquality(Data<int> const& set1, Data<int> const& set2){
	BOOST_REQUIRE_EQUAL(set1.numberOfBatches(),set2.numberOfBatches());
	BOOST_REQUIRE_EQUAL(set1.numberOfElements(),set2.numberOfElements());

	for(size_t i=0;i!=set1.numberOfBatches();++i) {
		IntVector vec1=set1.batch(i);
		IntVector vec2=set2.batch(i);
		BOOST_REQUIRE_EQUAL(vec1.size(),vec2.size());
		BOOST_CHECK_EQUAL_COLLECTIONS(vec1.begin(),vec1.end(),vec2.begin(),vec2.end());
	}
}

void testDatasetEquality(LabeledData<int, int> const& set1, LabeledData<int, int> const& set2){
	BOOST_REQUIRE_EQUAL(set1.numberOfBatches(),set2.numberOfBatches());
	BOOST_REQUIRE_EQUAL(set1.numberOfElements(),set2.numberOfElements());
	for(std::size_t i = 0; i != set1.numberOfBatches(); ++i){
		BOOST_REQUIRE_EQUAL(set1.batch(i).input.size(),set1.batch(i).label.size());
		BOOST_REQUIRE_EQUAL(set2.batch(i).input.size(),set2.batch(i).label.size());
	}
	testSetEquality(set1.inputs(),set2.inputs());
	testSetEquality(set1.labels(),set2.labels());
}


BOOST_AUTO_TEST_SUITE (Data_Dataset)

BOOST_AUTO_TEST_CASE( Set_Test )
{
	std::vector<int> inputs;

	// the test results
	std::vector<std::size_t> indizes[2];

	// fill the vectors: inputs are the number [100, ..., 199]
	for (int i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	for(std::size_t i = 0; i != 20; ++i){
		indizes[i%2].push_back(i);
	}
	
	// 1.1 test element access and thus createDataFromRange
	Data<int> set = createDataFromRange(inputs,5);//20 batches
	BOOST_REQUIRE_EQUAL(set.numberOfElements(), 100u);
	BOOST_CHECK_EQUAL(set.shape(), Shape({}));
	BOOST_REQUIRE_EQUAL(set.numberOfBatches(), 20u);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		set.elements().begin(),set.elements().end(),
		inputs.begin(),inputs.end()
	);
	
	//1.2 test batch access
	for (size_t i=0; i!=10; ++i) {
		IntVector batch=set.batch(i);
		BOOST_REQUIRE_EQUAL(batch.size(),5u);
		BOOST_CHECK_EQUAL_COLLECTIONS(
			batch.begin(),batch.end(),
			inputs.begin()+i*5,
			inputs.begin()+(i+1)*5
		);
	}
	
	// 2. create indexed partitions
	{
		Data<int> subset = set.indexedSubset(indizes[0]);
		BOOST_REQUIRE_EQUAL(subset.numberOfBatches(), indizes[0].size());

		for (size_t i=0; i!=subset.numberOfBatches(); ++i) {
			IntVector batch=subset.batch(i);
			BOOST_REQUIRE_EQUAL(batch.size(),5);
			BOOST_CHECK_EQUAL_COLLECTIONS(
				batch.begin(),batch.end(),
				inputs.begin()+indizes[0][i]*5,
				inputs.begin()+(indizes[0][i]+1)*5
			);
		}
	}
}

BOOST_AUTO_TEST_CASE( Set_Repartition )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (int i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	//generate a set and than repartition it with unevenly sized batches
	std::vector<std::size_t> batchSizes(8);
	batchSizes[0]=8;
	batchSizes[1]=24;
	batchSizes[2]=8;
	batchSizes[3]=7;
	batchSizes[4]=8;
	batchSizes[5]=12;
	batchSizes[6]=25;
	batchSizes[7]=8;
	Data<int> set = createDataFromRange(inputs,10);
	set.repartition(batchSizes);
	
	BOOST_REQUIRE_EQUAL(set.numberOfBatches(),8u);
	BOOST_REQUIRE_EQUAL(set.numberOfElements(),100u);
	for(std::size_t i = 0; i != 8; ++i){
		BOOST_CHECK_EQUAL(set.batch(i).size(), batchSizes[i]);
	}
	BOOST_CHECK_EQUAL_COLLECTIONS(
		set.elements().begin(),set.elements().end(),
		inputs.begin(),inputs.end()
	);
}

BOOST_AUTO_TEST_CASE( Set_splitAtElement_Boundary_Test )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (int i=0;i!=100;++i) {
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
	
	
	//split before and after every batch
	std::size_t index = 0;
	for(std::size_t i = 0; i <= batchSizes.size();++i){
		Data<int> set= createDataFromRange(inputs,10);
		set.repartition(batchSizes);
		Data<int> split = splitAtElement(set,index);
		
		BOOST_REQUIRE_EQUAL(set.numberOfBatches(),i);
		BOOST_REQUIRE_EQUAL(split.numberOfBatches(),8-i);
		BOOST_REQUIRE_EQUAL(set.numberOfElements(),index);
		BOOST_REQUIRE_EQUAL(split.numberOfElements(),100-index);
		
		BOOST_CHECK_EQUAL_COLLECTIONS(
			set.elements().begin(),set.elements().end(),
			inputs.begin(),inputs.begin()+index
		);
		BOOST_CHECK_EQUAL_COLLECTIONS(
			split.elements().begin(),split.elements().end(),
			inputs.begin()+index,inputs.end()
		);
		
		if(i != batchSizes.size())
			index+=batchSizes[i];
	}
}

BOOST_AUTO_TEST_CASE( Set_Merge_Test )
{
	std::vector<int> inputs1;
	std::vector<int> inputs2;

	for (int i=0;i!=50;++i) {
		inputs1.push_back(i);
	}
	
	for (int i=0;i!=70;++i) {
		inputs2.push_back(50+i);
	}
	
	Data<int> set1= createDataFromRange(inputs1,10);
	Data<int> set2= createDataFromRange(inputs2,7);
	
	set1.append(set2);
	
	BOOST_REQUIRE_EQUAL(set1.numberOfBatches(),15);
	BOOST_REQUIRE_EQUAL(set1.numberOfElements(),120);
	for(std::size_t i = 0; i != 15; ++i){
		if(i < 5){
			BOOST_CHECK_EQUAL(set1.batch(i).size(),10);
		}
		else{
			BOOST_CHECK_EQUAL(set1.batch(i).size(),7);
		}
	}

	for(std::size_t i = 0; i != 120;++i){
		BOOST_CHECK_EQUAL(set1.elements()[i],i);
	}
}

BOOST_AUTO_TEST_CASE( Data_ColumnAccess )
{
	std::vector<RealVector> inputs;
	RealVector test(50);
	for (size_t i=0;i!=50;++i) {
		RealVector r(2);
		r(0) = i / 2.0;
		r(1) = 5;
		inputs.push_back(r);
		test(i) = 0.1*i;
	}
	Data<RealVector> set = createDataFromRange(inputs);
	RealVector c0 = getColumn(set, 0);
	setColumn(set, 0, test);

	for(std::size_t i = 0; i != 50; ++i){
		BOOST_CHECK_EQUAL(set.elements()[i](0), test(i));
		BOOST_CHECK_EQUAL(c0(i), inputs[i](0));
	}
}


BOOST_AUTO_TEST_CASE( LabledData_Merge_Test )
{
	std::vector<int> inputs1;
	std::vector<int> labels1;
	std::vector<int> inputs2;
	std::vector<int> labels2;

	for (int i=0;i!=50;++i) {
		inputs1.push_back(i);
		labels1.push_back(i*2);
	}
	
	for (int i=0;i!=70;++i) {
		inputs2.push_back(50+i);
		labels2.push_back((50+i)*2);
	}
	
	LabeledData<int,int> set1= createLabeledDataFromRange(inputs1,labels1,10);
	LabeledData<int,int> set2= createLabeledDataFromRange(inputs2,labels2,7);
	
	set1.append(set2);
	
	BOOST_REQUIRE_EQUAL(set1.numberOfBatches(),15);
	BOOST_REQUIRE_EQUAL(set1.numberOfElements(),120);
	for(std::size_t i = 0; i != 15; ++i){
		if(i < 5){
			BOOST_CHECK_EQUAL(set1.batch(i).size(),10);
		}
		else{
			BOOST_CHECK_EQUAL(set1.batch(i).size(),7);
		}
	}

	for(std::size_t i = 0; i != 120;++i){
		BOOST_CHECK_EQUAL(set1.elements()[i].input,i);
		BOOST_CHECK_EQUAL(set1.elements()[i].label,2*i);
	}
}


BOOST_AUTO_TEST_CASE( Set_splitAtElement_MiddleOfBatch_Test )
{
	std::vector<int> inputs;

	// fill the vectors: inputs are the number [100, ..., 199]
	for (int i=0;i!=100;++i) {
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
	
	
	//split in the middle of a batch
	Data<int> set= createDataFromRange(inputs,10);
	set.repartition(batchSizes);
	Data<int> split = splitAtElement(set,53);
	
	BOOST_REQUIRE_EQUAL(set.numberOfBatches(),5u);
	BOOST_REQUIRE_EQUAL(split.numberOfBatches(),4u);
	BOOST_REQUIRE_EQUAL(set.numberOfElements(),53u);
	BOOST_REQUIRE_EQUAL(split.numberOfElements(),47u);
	
	BOOST_CHECK_EQUAL_COLLECTIONS(
		set.elements().begin(),set.elements().end(),
		inputs.begin(),inputs.begin()+53
	);
	BOOST_CHECK_EQUAL_COLLECTIONS(
		split.elements().begin(),split.elements().end(),
		inputs.begin()+53,inputs.end()
	);
}


BOOST_AUTO_TEST_CASE( RepartitionByClass_Test )
{
	std::vector<UIntVector> inputs(101,UIntVector(3));
	std::vector<unsigned int> labels(101);

	// generate dataset
	for (int i=0;i!=101;++i) {
		inputs[i][0] = 100+i;
		inputs[i][1] = 200+i;
		inputs[i][2] = 300+i;
		labels[i] = i%3;
	}
	LabeledData<UIntVector,unsigned int> data = createLabeledDataFromRange(inputs,labels,9);
	
	BOOST_REQUIRE_EQUAL(data.numberOfElements(),101);
	BOOST_REQUIRE_EQUAL(data.numberOfBatches(),12);
	
	repartitionByClass(data,11);//different batch size to check other side effects
	//check dataset integrity
	BOOST_REQUIRE_EQUAL(data.numberOfElements(),101);
	BOOST_REQUIRE_EQUAL(data.numberOfBatches(),11);
	std::vector<std::size_t> classes = classSizes(data);
	BOOST_REQUIRE_EQUAL(classes[0],34);
	BOOST_REQUIRE_EQUAL(classes[1],34);
	BOOST_REQUIRE_EQUAL(classes[2],33);
	
	//check that all labels match the elements and that all elements are still there
	std::vector<unsigned int> resultInputs(101,0);
	for(std::size_t i = 0; i != 101; ++i){
		std::size_t k = data.elements()[i].input(0)-100;
		BOOST_CHECK_EQUAL(data.elements()[i].input(1),k+200);
		BOOST_CHECK_EQUAL(data.elements()[i].input(2),k+300);
		BOOST_CHECK_EQUAL(data.elements()[i].label,k%3);
		resultInputs[k] = (unsigned int)k;
	}
	//in the end all elements should be set
	for(std::size_t i = 0; i != 101; ++i){
		BOOST_CHECK_EQUAL(resultInputs[i],i);
	}
	
	//check the correct sizes of the batches
	BOOST_CHECK_EQUAL(data.batch(0).size(),9);
	BOOST_CHECK_EQUAL(data.batch(1).size(),9);
	BOOST_CHECK_EQUAL(data.batch(2).size(),8);
	BOOST_CHECK_EQUAL(data.batch(3).size(),8);
	BOOST_CHECK_EQUAL(data.batch(4).size(),9);
	BOOST_CHECK_EQUAL(data.batch(5).size(),9);
	BOOST_CHECK_EQUAL(data.batch(6).size(),8);
	BOOST_CHECK_EQUAL(data.batch(7).size(),8);
	BOOST_CHECK_EQUAL(data.batch(8).size(),11);
	BOOST_CHECK_EQUAL(data.batch(9).size(),11);
	BOOST_CHECK_EQUAL(data.batch(10).size(),11);
	
	//check order of the labels of the elements
	for(std::size_t i = 0; i != 34; ++i){
		BOOST_CHECK_EQUAL(data.elements()[i].label, 0);
	}
	for(std::size_t i = 34; i != 68; ++i){
		BOOST_CHECK_EQUAL(data.elements()[i].label, 1);
	}
	for(std::size_t i = 68; i != 101; ++i){
		BOOST_CHECK_EQUAL(data.elements()[i].label, 2);
	}
}

BOOST_AUTO_TEST_CASE( BinarySubproblem_Test )
{
	std::vector<unsigned int> labels ={0,0,1,1,1,2,2,3,4,4,4};
	std::vector<std::size_t> batch_sizes={10,11,12,13,14,15,16,17,18,19,20};
	std::vector<std::size_t> sizes={21,39,31,17,57};
	LabeledData<UIntVector,unsigned int> data(batch_sizes,{1,5});
	for(std::size_t i = 0; i != 11; ++i){
		for(std::size_t j = 0; j != batch_sizes[i]; ++j){
			data.batch(i).input(j,0) = (unsigned int)j;
			data.batch(i).label(j) = labels[i];
		}
	}
	
	//check all class combinations
	for(unsigned int ci = 0; ci != 5; ++ci){
		for(unsigned int cj = 0; cj != 5; ++cj){
			if(ci == cj) continue;
			LabeledData<UIntVector,unsigned int> testset = binarySubProblem(data,ci,cj);
			BOOST_REQUIRE_EQUAL(numberOfClasses(testset),2);
			std::vector<std::size_t> classes = classSizes(testset);
			BOOST_CHECK_EQUAL(classes[0], sizes[ci]);
			BOOST_CHECK_EQUAL(classes[1], sizes[cj]);
		}
	}
	
	
}



// 	// 2.2 range version
// 	std::cout << "range...";
// 	Data<int> rangeSubset, rangeSubset2;
// 	Data<int> rangeComplement;
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
// 	Data<int> randomSubset;
// 	Data<int> randomComplement;
// 	set.randomSubset(12, randomSubset, randomComplement);
// 	BOOST_REQUIRE_EQUAL(randomSubset.size(), 12);
// 	BOOST_REQUIRE_EQUAL(randomComplement.size(), 8);
// 
// 	// 2.4 subsets from subsets
// 	std::cout << "subsubset...";
// 	Data<int> subsubset;
// 	subset.rangeSubset(5, subsubset);
// 	BOOST_REQUIRE_EQUAL(subsubset.size(), 5);
// 	for(size_t i=0; i!=subsubset.size(); ++i) {
// 		BOOST_CHECK_EQUAL(indizes[0][i] + 20, subsubset(i));
// 	}
// 
// 	// 2.5 packing sets and checking equality
// 	//2.5.1 equality
// 	std::cout<<"equal1...";
// 	Data<int> subset3 = subset2;
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
// 	Data<int> namedSet;
// 	set.namedSubset("Dataset_test", namedSet);
// 	testSetEquality(namedSet, subset);
// 	BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);
// 
// 	// 3.1 now with complement
// 	std::cout << "subset complement...";
// 	Data<int> namedComplement;
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
// 	Data<int> namedSet2;
// 	set.namedSubset("Dataset_test2",namedSet2);
// 	testSetEquality(namedSet2,randomSubset);
// 
// 
// 	// 4. copy constructor, default constructor and equality
// 	// a bit late, but...
// 	std::cout << "copy...";
// 	Data<int> copy(set);
// 	testSetEquality(copy, set);
// 	BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), true);
// 	Data<int> defaultSet;
// 	BOOST_REQUIRE_EQUAL(defaultSet.size(), 0);
// 	copy = defaultSet;
// 	BOOST_REQUIRE_EQUAL(copy.size(), 0);
// 	BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), false);

	//~ // 2.4 subsets from subsets
	//~ std::cout << "subsubset...";
	//~ Data<int> subsubset;
	//~ subset.rangeSubset(5, subsubset);
	//~ BOOST_REQUIRE_EQUAL(subsubset.size(), 5);
	//~ for(size_t i=0; i!=subsubset.size(); ++i) {
		//~ BOOST_CHECK_EQUAL(indizes[0][i] + 20, subsubset(i));
	//~ }

	//~ // 2.5 packing sets and checking equality
	//~ //2.5.1 equality
	//~ std::cout<<"equal1...";
	//~ Data<int> subset3 = subset2;
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

	//~ Data<int> namedSet;
	//~ set.namedSubset("Dataset_test", namedSet);
	//~ testSetEquality(namedSet, subset);
	//~ BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);

	//~ // 3.1 now with complement
	//~ std::cout << "subset complement...";
	//~ Data<int> namedComplement;
	//~ set.namedSubset("Dataset_test", namedSet, namedComplement);
	//~ testSetEquality(namedSet, subset);
	//~ testSetEquality(namedComplement, complement);
	//~ BOOST_REQUIRE_EQUAL(namedSet.hasNamedSubset("Dataset_test"), false);
	//~ BOOST_REQUIRE_EQUAL(namedComplement.hasNamedSubset("Dataset_test"), false);

	//~ //3.2 create from DataSet
	//~ std::cout<<"SetSubset...";
	//~ set.createNamedSubset("Dataset_test2",randomSubset);
	//~ BOOST_REQUIRE_EQUAL(set.hasNamedSubset("Dataset_test2"),true);

	//~ Data<int> namedSet2;
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
	//~ Data<int> copy(set);
	//~ testSetEquality(copy, set);
	//~ BOOST_REQUIRE_EQUAL(copy.hasNamedSubset("Dataset_test"), true);
	//~ Data<int> defaultSet;
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
	Data<int> inputSet=set.inputs();
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

BOOST_AUTO_TEST_CASE( DATA_SERIALIZE )
{
	std::vector<int> data(1000);
	for (int i=0; i<1000; i++){
		data[i]=3*i+5;
	}
	Data<int> dataSource = createDataFromRange(data,23);

	//now we serialize the Dataset
	std::ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << dataSource;

	//and create a new set from the serialization
	Data<int> dataDeserialized;
	std::istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> dataDeserialized;
	
	testSetEquality(dataSource,dataDeserialized);
}

BOOST_AUTO_TEST_CASE( LABELED_DATA_SERIALIZE )
{
	std::vector<int> data(1000);
	std::vector<int> labels(1000);
	for (int i=0; i<1000; i++){
		data[i]=3*i+5;
		labels[i]=5*i+1001;
	}
	LabeledData<int,int> dataSource = createLabeledDataFromRange(data,labels,23);

	//now we serialize the Dataset
	std::ostringstream outputStream;  
	TextOutArchive oa(outputStream);  
	oa << dataSource;

	//and create a new set from the serialization
	LabeledData<int,int> dataDeserialized;
	std::istringstream inputStream(outputStream.str());  
	TextInArchive ia(inputStream);
	ia >> dataDeserialized;
	
	testDatasetEquality(dataSource,dataDeserialized);
}


BOOST_AUTO_TEST_SUITE_END()
