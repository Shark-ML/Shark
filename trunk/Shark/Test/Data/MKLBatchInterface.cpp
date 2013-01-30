#define BOOST_TEST_MODULE Data_MKLBatchInterface
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

#include <shark/Data/MKLBatchInterface.h>

using namespace shark;


//template<class T, class U>
//struct SameReference;

//template<class TestTuple, class Get,class Tuple>
//struct SameReference<Tuple,detail::FusionVectorBatchReference<TestTuple,Get> >
//: public boost::is_same<TestTuple,Tuple>{};

////100% of this test is done during compile time...
//BOOST_AUTO_TEST_CASE( MKL_Interface_Test_COMPILE )
//{
//	//the test type.
//	typedef boost::fusion::vector<double,RealVector, boost::fusion::vector<std::string,IntVector> > Tuple;
//	
//	//test1: is the Batch type correct?
//	typedef boost::fusion::vector3<RealVector,RealMatrix, boost::fusion::vector2<std::vector<std::string>,IntMatrix> > BatchTuple;
//	typedef Batch<Tuple>::type BatchTupleTest;
//	BOOST_STATIC_ASSERT((boost::is_same<BatchTuple,BatchTupleTest>::value));
//	
//	//test2: is the reference type correct?
//	typedef boost::fusion::vector2<std::string&,Batch<IntVector>::reference> InnerBatchReference;
//	typedef Batch<boost::fusion::vector<std::string,IntVector> >::reference InnerBatchReferenceTest;
//	BOOST_STATIC_ASSERT((SameReference<InnerBatchReference,InnerBatchReferenceTest>::value));
//	
//	typedef boost::fusion::vector3<double&,Batch<RealVector>::reference, InnerBatchReferenceTest > BatchReference;
//	typedef Batch<Tuple>::reference BatchReferenceTest;
//	BOOST_STATIC_ASSERT((SameReference<BatchReference,BatchReferenceTest>::value));
//	
//	//test3: is the const_reference type correct?
//	typedef boost::fusion::vector2<std::string const&,Batch<IntVector>::const_reference> InnerBatchConstReference;
//	typedef Batch<boost::fusion::vector<std::string,IntVector> >::const_reference InnerBatchConstReferenceTest;
//	BOOST_STATIC_ASSERT((SameReference<InnerBatchConstReference,InnerBatchConstReferenceTest>::value));
//	
//	typedef boost::fusion::vector3<double const&,Batch<RealVector>::const_reference, InnerBatchConstReferenceTest > BatchConstReference;
//	typedef Batch<Tuple>::const_reference BatchConstReferenceTest;
//	BOOST_STATIC_ASSERT((SameReference<BatchConstReference,BatchConstReferenceTest>::value));
//	
//}

//class Get
//{
//private:
//	std::size_t m_index;
//public:
//	Get(std::size_t index):m_index(index){}
//	
//	template<typename Sig>
//	struct result;

//	template<typename U>
//	struct result<Get(U const&)>{
//		typedef U& type;
//	};

//	template <typename U>
//	typename result<Get(U const&)>::type operator()(U const& x) const
//	{
//		//fusion does not allow x to be non-const
//		//return get(const_cast<U&>(x),m_index);
//		return const_cast<U&>(x);
//	}
//};

BOOST_AUTO_TEST_CASE( MKL_Interface_Test_References )
{
	//the used types.
	typedef boost::fusion::vector<unsigned int,UIntVector > Tuple;
	typedef boost::fusion::vector<UIntVector, UIntMatrix > BatchTuple;
	typedef Batch<Tuple>::reference BatchReference;
	typedef Batch<Tuple>::const_reference BatchConstReference;
	
	//initialize batch
	BatchTuple batch(RealVector(10),RealMatrix(10,5));
	for(std::size_t i  = 0; i != 10; ++i){
		boost::fusion::at_c<0>(batch)(i)=i;//vector
		for(std::size_t j  = 0; j != 5; ++j){//matrix
			boost::fusion::at_c<1>(batch)(i,j)=i*5+j;
		}
	}
	
	//test whether the references produce the correct elements of the batch
	for(std::size_t i  = 0; i != 10; ++i){
		BatchReference ref(batch,i);
		BatchConstReference constRef(batch,i);
		BOOST_CHECK_EQUAL(boost::fusion::at_c<0>(ref),i);
		BOOST_CHECK_EQUAL(boost::fusion::at_c<0>(constRef),i);
		for(std::size_t j  = 0; j != 5; ++j){//matrix
			BOOST_CHECK_EQUAL(boost::fusion::at_c<1>(ref)(j),i*5+j);
			BOOST_CHECK_EQUAL(boost::fusion::at_c<1>(constRef)(j),i*5+j);
		}
	}
	
//	boost::fusion::vector<int, double> a(1,2.1);
//	boost::fusion::vector<int&, double&> b(boost::fusion::transform(a,Get(0)));
	
}
