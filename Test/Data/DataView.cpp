#define BOOST_TEST_MODULE Data_DataView
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/DataView.h>
#include <shark/Data/Dataset.h>

using namespace shark;


BOOST_AUTO_TEST_SUITE (Data_DataView)


int get(Data<int> const& data, std::size_t index){
	std::size_t pos = 0;
	std::size_t b = 0;
	for(; pos + data[b].size() <= index; ++b){
		pos += data[b].size();
	}
	return data[b](index - pos);
}


InputLabelPair<int, unsigned int> get(LabeledData<int,unsigned int> const& data, std::size_t index){
	std::size_t pos = 0;
	std::size_t b = 0;
	for(; pos + data[b].size() <= index; ++b){
		pos += data[b].size();
	}
	return getBatchElement(data[b],index - pos);
}

typedef boost::mpl::list<Data<int>, Data<int> const> SetTypes;
typedef boost::mpl::list<LabeledData<int,unsigned int>, LabeledData<int,unsigned int> const> LabeledSetTypes;
BOOST_AUTO_TEST_CASE_TEMPLATE(DataView_Data_Test, SetType,SetTypes){
	//define initial dataset
	std::vector<int> inputs;
	for (int i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	Data<int> set0 =  createDataFromRange(inputs,10);
	SetType& set = set0; 
	
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
	DataView<SetType > view(set);
	BOOST_CHECK_EQUAL(view.size(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(view[i],get(set,i));
		BOOST_CHECK_EQUAL(view.index(i), i);
	}
	
	//Test2 create a subset of the identity set
	DataView<SetType > subview1=subset(view,subsetIndices1);
	BOOST_CHECK_EQUAL(subview1.size(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(subview1[i], get(set,subsetIndices1[i]));
		BOOST_CHECK_EQUAL(subview1.index(i), subsetIndices1[i]);
	}
	
	//Test3 create a subset of the subset
	DataView<SetType > subview2=subset(subview1,subsetIndices2);
	BOOST_CHECK_EQUAL(subview2.size(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(subview2[i],get(set,realSubsetIndices2[i]));
		BOOST_CHECK_EQUAL(subview2.index(i), realSubsetIndices2[i]);
	}
	
	//Test4 create a Dataset of the sets
	Data<int> setCopy= toDataset(view,20);
	BOOST_CHECK_EQUAL(setCopy.size(),5u);
	BOOST_CHECK_EQUAL(setCopy.numberOfElements(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(get(setCopy,i),get(set,i));
	}
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK_EQUAL(setCopy[i].size(),20u);
	}
	
	Data<int> copy1 = toDataset(subview1,5);
	BOOST_CHECK_EQUAL(copy1.size(),6u);
	BOOST_CHECK_EQUAL(copy1.numberOfElements(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(get(copy1,i), get(set,subsetIndices1[i]));
	}
	for(std::size_t i = 0; i != 6; ++i){
		BOOST_CHECK_EQUAL(copy1[i].size(),5u);
	}
	
	Data<int> copy2 = toDataset(subview2,2);
	BOOST_CHECK_EQUAL(copy2.size(),5u);
	BOOST_CHECK_EQUAL(copy2.numberOfElements(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(get(copy2,i),get(set,realSubsetIndices2[i]));
	}
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(copy2[i].size(),2u);
	}
}

//now the same test for datasets


BOOST_AUTO_TEST_CASE_TEMPLATE(DataView_Dataset_Test, SetType,LabeledSetTypes){
	//define initial dataset
	std::vector<int> inputs;
	for (int i=0;i!=100;++i) {
		inputs.push_back(100+i);
	}
	std::vector<unsigned int> labels;
	for (unsigned int i=0;i!=100;++i) {
		labels.push_back(200+i);
	}
	LabeledData<int,unsigned int> set0 = createLabeledDataFromRange(inputs,labels,10);
	SetType& set = set0;
	
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
	DataView<SetType> view(set);
	BOOST_CHECK_EQUAL(view.size(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(view[i].input, get(set, i).input);
		BOOST_CHECK_EQUAL(view[i].label,get(set, i).label);
		BOOST_CHECK_EQUAL(view.index(i), i);
	}
	
	//Test2 create a subset of the identity set
	DataView<SetType> subview1=subset(view,subsetIndices1);
	BOOST_CHECK_EQUAL(subview1.size(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(subview1[i].input,get(set, subsetIndices1[i]).input);
		BOOST_CHECK_EQUAL(subview1[i].label,get(set, subsetIndices1[i]).label);
		BOOST_CHECK_EQUAL(subview1.index(i), subsetIndices1[i]);
	}
	
	//Test3 create a subset of the subset
	DataView<SetType> subview2=subset(subview1,subsetIndices2);
	BOOST_CHECK_EQUAL(subview2.size(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(subview2[i].input,get(set, realSubsetIndices2[i]).input);
		BOOST_CHECK_EQUAL(subview2[i].label,get(set, realSubsetIndices2[i]).label);
		BOOST_CHECK_EQUAL(subview2.index(i), realSubsetIndices2[i]);
	}
	
	//Test4 create a Dataset of the sets
	LabeledData<int,unsigned int> setCopy= toDataset(view,20);
	BOOST_CHECK_EQUAL(setCopy.size(),5u);
	BOOST_CHECK_EQUAL(setCopy.numberOfElements(),100u);
	for(std::size_t i = 0; i != 100; ++i){
		BOOST_CHECK_EQUAL(get(setCopy,i).input,get(set, i).input);
		BOOST_CHECK_EQUAL(get(setCopy,i).label,get(set, i).label);
	}
	for(std::size_t i = 0; i != 5; ++i){
		BOOST_CHECK_EQUAL(setCopy[i].size(),20u);
	}
	
	LabeledData<int,unsigned int> copy1 = toDataset(subview1,5);
	BOOST_CHECK_EQUAL(copy1.size(),6u);
	BOOST_CHECK_EQUAL(copy1.numberOfElements(),30u);
	for(std::size_t i = 0; i != 30; ++i){
		BOOST_CHECK_EQUAL(get(copy1,i).input,get(set, subsetIndices1[i]).input);
		BOOST_CHECK_EQUAL(get(copy1,i).label,get(set, subsetIndices1[i]).label);
	}
	for(std::size_t i = 0; i != 6; ++i){
		BOOST_CHECK_EQUAL(copy1[i].size(),5u);
	}
	
	LabeledData<int,unsigned int> copy2 = toDataset(subview2,2);
	BOOST_CHECK_EQUAL(copy2.size(),5u);
	BOOST_CHECK_EQUAL(copy2.numberOfElements(),10u);
	for(std::size_t i = 0; i != 10; ++i){
		BOOST_CHECK_EQUAL(get(copy2,i).input,get(set, realSubsetIndices2[i]).input);
		BOOST_CHECK_EQUAL(get(copy2,i).label,get(set, realSubsetIndices2[i]).label);
	}
	for(std::size_t i = 0; i != 5;++i){
		BOOST_CHECK_EQUAL(copy2[i].size(),2u);
	}
}


BOOST_AUTO_TEST_SUITE_END()
