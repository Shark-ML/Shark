#define BOOST_TEST_MODULE Data_DataView
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/DataView.h>

using namespace shark;


BOOST_AUTO_TEST_CASE( DataView_Data_Test )
{
	//define initial dataset
	std::vector<int> inputs;
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	UnlabeledData<int> set(inputs,10);
	
	//define subset
	std::vector<std::size_t> subsetIndices1;
	for (size_t i=0;i!=30;++i) {
		subsetIndices1.push_back(i*3+1);
	}
	
	//define indices for a sub-subset
	std::vector<std::size_t> subsetIndices2;
	std::vector<std::size_t> realSubsetIndices2;
	for (size_t i=0;i!=10;++i) {
		subsetIndices2.push_back(2*i+3);
		realSubsetIndices2.push_back(subsetIndices1[2*i+3]);
	}
	
	//Test1 create just an identity set
	DataView<Data<int> > view(set);
	BOOST_CHECK_EQUAL(view.size(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(view[i],set(i));
		BOOST_CHECK_EQUAL(view.index(i), i);
	}
	
	//Test2 create a subset of the identity set
	DataView<Data<int> > subview1=subset(view,subsetIndices1);
	BOOST_CHECK_EQUAL(subview1.size(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(subview1[i],set(subsetIndices1[i]));
		BOOST_CHECK_EQUAL(subview1.index(i), subsetIndices1[i]);
	}
	
	//Test3 create a subset of the subset
	DataView<Data<int> > subview2=subset(subview1,subsetIndices2);
	BOOST_CHECK_EQUAL(subview2.size(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(subview2[i],set(realSubsetIndices2[i]));
		BOOST_CHECK_EQUAL(subview2.index(i), realSubsetIndices2[i]);
	}
	
	//Test4 create a Dataset of the sets
	Data<int> setCopy= toDataset(view,10);
	BOOST_CHECK_EQUAL(setCopy.size(),10u);
	BOOST_CHECK_EQUAL(setCopy.numberOfElements(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(setCopy(i),set(i));
	}
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(setCopy.batch(i).size(),10u);
	}
	
	Data<int> copy1 = toDataset(subview1,5);
	BOOST_CHECK_EQUAL(copy1.size(),6u);
	BOOST_CHECK_EQUAL(copy1.numberOfElements(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(copy1(i),set(subsetIndices1[i]));
	}
	for(std::size_t i = 0; i != 6; ++i){
		BOOST_CHECK_EQUAL(copy1.batch(i).size(),5u);
	}
	
	Data<int> copy2 = toDataset(subview2,2);
	BOOST_CHECK_EQUAL(copy2.size(),5u);
	BOOST_CHECK_EQUAL(copy2.numberOfElements(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(copy2(i),set(realSubsetIndices2[i]));
	}
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(copy2.batch(i).size(),2u);
	}
}

//Same again, just throwing in some consts...
BOOST_AUTO_TEST_CASE( DataView_Data_Const_Test )
{
	//define initial dataset
	std::vector<int> inputs;
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	UnlabeledData<int> set0(inputs,10);
	UnlabeledData<int> const& set = set0;
	
	//define subset
	std::vector<std::size_t> subsetIndices1;
	for (size_t i=0;i!=30;++i) {
		subsetIndices1.push_back(i*3+1);
	}
	
	//define indices for a sub-subset
	std::vector<std::size_t> subsetIndices2;
	std::vector<std::size_t> realSubsetIndices2;
	for (size_t i=0;i!=10;++i) {
		subsetIndices2.push_back(2*i+3);
		realSubsetIndices2.push_back(subsetIndices1[2*i+3]);
	}
	
	//Test1 create just an identity set
	DataView<Data<int> const > view(set);
	BOOST_CHECK_EQUAL(view.size(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(view[i],set(i));
		BOOST_CHECK_EQUAL(view.index(i), i);
	}
	
	//Test2 create a subset of the identity set
	DataView<Data<int> const> subview1=subset(view,subsetIndices1);
	BOOST_CHECK_EQUAL(subview1.size(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(subview1[i],set(subsetIndices1[i]));
		BOOST_CHECK_EQUAL(subview1.index(i), subsetIndices1[i]);
	}
	
	//Test3 create a subset of the subset
	DataView<Data<int> const> subview2=subset(subview1,subsetIndices2);
	BOOST_CHECK_EQUAL(subview2.size(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(subview2[i],set(realSubsetIndices2[i]));
		BOOST_CHECK_EQUAL(subview2.index(i), realSubsetIndices2[i]);
	}
	
	//Test4 create a Dataset of the sets
	Data<int> setCopy= toDataset(view,10);
	BOOST_CHECK_EQUAL(setCopy.size(),10u);
	BOOST_CHECK_EQUAL(setCopy.numberOfElements(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(setCopy(i),set(i));
	}
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(setCopy.batch(i).size(),10u);
	}
	
	Data<int> copy1 = toDataset(subview1,5);
	BOOST_CHECK_EQUAL(copy1.size(),6u);
	BOOST_CHECK_EQUAL(copy1.numberOfElements(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(copy1(i),set(subsetIndices1[i]));
	}
	for(std::size_t i = 0; i != 6; ++i){
		BOOST_CHECK_EQUAL(copy1.batch(i).size(),5u);
	}
	
	Data<int> copy2 = toDataset(subview2,2);
	BOOST_CHECK_EQUAL(copy2.size(),5u);
	BOOST_CHECK_EQUAL(copy2.numberOfElements(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(copy2(i),set(realSubsetIndices2[i]));
	}
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(copy2.batch(i).size(),2u);
	}
}

//now the same test for datasets

BOOST_AUTO_TEST_CASE( DataView_Dataset_Test )
{
	//define initial dataset
	std::vector<int> inputs;
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	std::vector<unsigned int> labels;
	for (size_t i=0;i!=100;++i) {
		labels.push_back(200+i);
	}
	LabeledData<int,unsigned int> set(inputs,labels,10);
	
	//define subset
	std::vector<std::size_t> subsetIndices1;
	for (size_t i=0;i!=30;++i) {
		subsetIndices1.push_back(i*3+1);
	}
	
	//define indices for a sub-subset
	std::vector<std::size_t> subsetIndices2;
	std::vector<std::size_t> realSubsetIndices2;
	for (size_t i=0;i!=10;++i) {
		subsetIndices2.push_back(2*i+3);
		realSubsetIndices2.push_back(subsetIndices1[2*i+3]);
	}
	
	//Test1 create just an identity set
	DataView<LabeledData<int,unsigned int> > view(set);
	BOOST_CHECK_EQUAL(view.size(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(view[i].input,set(i).input);
		BOOST_CHECK_EQUAL(view[i].label,set(i).label);
		BOOST_CHECK_EQUAL(view.index(i), i);
	}
	
	//Test2 create a subset of the identity set
	DataView<LabeledData<int,unsigned int> > subview1=subset(view,subsetIndices1);
	BOOST_CHECK_EQUAL(subview1.size(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(subview1[i].input,set(subsetIndices1[i]).input);
		BOOST_CHECK_EQUAL(subview1[i].label,set(subsetIndices1[i]).label);
		BOOST_CHECK_EQUAL(subview1.index(i), subsetIndices1[i]);
	}
	
	//Test3 create a subset of the subset
	DataView<LabeledData<int,unsigned int> > subview2=subset(subview1,subsetIndices2);
	BOOST_CHECK_EQUAL(subview2.size(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(subview2[i].input,set(realSubsetIndices2[i]).input);
		BOOST_CHECK_EQUAL(subview2[i].label,set(realSubsetIndices2[i]).label);
		BOOST_CHECK_EQUAL(subview2.index(i), realSubsetIndices2[i]);
	}
	
	//Test4 create a Dataset of the sets
	LabeledData<int,unsigned int> setCopy= toDataset(view,10);
	BOOST_CHECK_EQUAL(setCopy.size(),10u);
	BOOST_CHECK_EQUAL(setCopy.numberOfElements(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(setCopy(i).input,set(i).input);
		BOOST_CHECK_EQUAL(setCopy(i).label,set(i).label);
	}
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(setCopy.batch(i).size(),10u);
	}
	
	LabeledData<int,unsigned int> copy1 = toDataset(subview1,5);
	BOOST_CHECK_EQUAL(copy1.size(),6u);
	BOOST_CHECK_EQUAL(copy1.numberOfElements(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(copy1(i).input,set(subsetIndices1[i]).input);
		BOOST_CHECK_EQUAL(copy1(i).label,set(subsetIndices1[i]).label);
	}
	for(std::size_t i = 0; i != 6; ++i){
		BOOST_CHECK_EQUAL(copy1.batch(i).size(),5u);
	}
	
	LabeledData<int,unsigned int> copy2 = toDataset(subview2,2);
	BOOST_CHECK_EQUAL(copy2.size(),5u);
	BOOST_CHECK_EQUAL(copy2.numberOfElements(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(copy2(i).input,set(realSubsetIndices2[i]).input);
		BOOST_CHECK_EQUAL(copy2(i).label,set(realSubsetIndices2[i]).label);
	}
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(copy2.batch(i).size(),2u);
	}
}


//now the same test for const datasets

BOOST_AUTO_TEST_CASE( DataView_Dataset_Const_Test )
{
	//define initial dataset
	std::vector<int> inputs;
	for (size_t i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	std::vector<unsigned int> labels;
	for (size_t i=0;i!=100;++i) {
		labels.push_back(200+i);
	}
	LabeledData<int,unsigned int> set0(inputs,labels,10);
	LabeledData<int,unsigned int> const& set = set0;
	//define subset
	std::vector<std::size_t> subsetIndices1;
	for (size_t i=0;i!=30;++i) {
		subsetIndices1.push_back(i*3+1);
	}
	
	//define indices for a sub-subset
	std::vector<std::size_t> subsetIndices2;
	std::vector<std::size_t> realSubsetIndices2;
	for (size_t i=0;i!=10;++i) {
		subsetIndices2.push_back(2*i+3);
		realSubsetIndices2.push_back(subsetIndices1[2*i+3]);
	}
	
	//Test1 create just an identity set
	DataView<LabeledData<int,unsigned int> const > view(set);
	BOOST_CHECK_EQUAL(view.size(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(view[i].input,set(i).input);
		BOOST_CHECK_EQUAL(view[i].label,set(i).label);
		BOOST_CHECK_EQUAL(view.index(i), i);
	}
	
	//Test2 create a subset of the identity set
	DataView<LabeledData<int,unsigned int> const > subview1=subset(view,subsetIndices1);
	BOOST_CHECK_EQUAL(subview1.size(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(subview1[i].input,set(subsetIndices1[i]).input);
		BOOST_CHECK_EQUAL(subview1[i].label,set(subsetIndices1[i]).label);
		BOOST_CHECK_EQUAL(subview1.index(i), subsetIndices1[i]);
	}
	
	//Test3 create a subset of the subset
	DataView<LabeledData<int,unsigned int> const > subview2=subset(subview1,subsetIndices2);
	BOOST_CHECK_EQUAL(subview2.size(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(subview2[i].input,set(realSubsetIndices2[i]).input);
		BOOST_CHECK_EQUAL(subview2[i].label,set(realSubsetIndices2[i]).label);
		BOOST_CHECK_EQUAL(subview2.index(i), realSubsetIndices2[i]);
	}
	
	//Test4 create a Dataset of the sets
	LabeledData<int,unsigned int> setCopy= toDataset(view,10);
	BOOST_CHECK_EQUAL(setCopy.size(),10u);
	BOOST_CHECK_EQUAL(setCopy.numberOfElements(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(setCopy(i).input,set(i).input);
		BOOST_CHECK_EQUAL(setCopy(i).label,set(i).label);
	}
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(setCopy.batch(i).size(),10u);
	}
	
	LabeledData<int,unsigned int> copy1 = toDataset(subview1,5);
	BOOST_CHECK_EQUAL(copy1.size(),6u);
	BOOST_CHECK_EQUAL(copy1.numberOfElements(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(copy1(i).input,set(subsetIndices1[i]).input);
		BOOST_CHECK_EQUAL(copy1(i).label,set(subsetIndices1[i]).label);
	}
	for(std::size_t i = 0; i != 6; ++i){
		BOOST_CHECK_EQUAL(copy1.batch(i).size(),5u);
	}
	
	LabeledData<int,unsigned int> copy2 = toDataset(subview2,2);
	BOOST_CHECK_EQUAL(copy2.size(),5u);
	BOOST_CHECK_EQUAL(copy2.numberOfElements(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(copy2(i).input,set(realSubsetIndices2[i]).input);
		BOOST_CHECK_EQUAL(copy2(i).label,set(realSubsetIndices2[i]).label);
	}
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(copy2.batch(i).size(),2u);
	}
}