#define BOOST_TEST_MODULE LinAlg_MatrixProxy
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;

template<class M1, class M2>
void checkDenseMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	//indexed access
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
	//iterator access rows
	for(std::size_t i = 0; i != m2.size1(); ++i){
		typedef typename M1::const_row_iterator Iter;
		BOOST_REQUIRE_EQUAL(m1.row_end(i)-m1.row_begin(i), m1.size2());
		std::size_t k = 0;
		for(Iter it = m1.row_begin(i); it != m1.row_end(i); ++it,++k){
			BOOST_CHECK_EQUAL(k,it.index());
			BOOST_CHECK_EQUAL(*it,m2(i,k));
		}
		//test that the actual iterated length equals the number of elements
		BOOST_CHECK_EQUAL(k, m1.size2());
	}
	//iterator access columns
	for(std::size_t i = 0; i != m2.size2(); ++i){
		typedef typename M1::const_column_iterator Iter;
		BOOST_REQUIRE_EQUAL(m1.column_end(i)-m1.column_begin(i), m1.size1());
		std::size_t k = 0;
		for(Iter it = m1.column_begin(i); it != m1.column_end(i); ++it,++k){
			BOOST_CHECK_EQUAL(k,it.index());
			BOOST_CHECK_EQUAL(*it,m2(k,i));
		}
		//test that the actual iterated length equals the number of elements
		BOOST_CHECK_EQUAL(k, m1.size1());
	}
}
template<class M1, class M2>
void checkDenseMatrixAssignment(M1& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	//indexed access
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			m1(i,j) = 0;
			BOOST_CHECK_EQUAL(m1(i,j),0);
			m1(i,j) = m2(i,j);
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
			m1(i,j) = 0;
			BOOST_CHECK_EQUAL(m1(i,j),0);
		}
	}
	//iterator access rows
	for(std::size_t i = 0; i != m2.size1(); ++i){
		typedef typename M1::row_iterator Iter;
		BOOST_REQUIRE_EQUAL(m1.row_end(i)-m1.row_begin(i), m1.size2());
		std::size_t k = 0;
		for(Iter it = m1.row_begin(i); it != m1.row_end(i); ++it,++k){
			BOOST_CHECK_EQUAL(k,it.index());
			*it=0;
			BOOST_CHECK_EQUAL(*it,0);
			BOOST_CHECK_EQUAL(m1(i,k),0);
			*it = m2(i,k);
			BOOST_CHECK_EQUAL(*it,m2(i,k));
			BOOST_CHECK_EQUAL(m1(i,k),m2(i,k));
			*it=0;
			BOOST_CHECK_EQUAL(*it,0);
			BOOST_CHECK_EQUAL(m1(i,k),0);
		}
		//test that the actual iterated length equals the number of elements
		BOOST_CHECK_EQUAL(k, m1.size2());
	}
	//iterator access columns
	for(std::size_t i = 0; i != m2.size2(); ++i){
		typedef typename M1::column_iterator Iter;
		BOOST_REQUIRE_EQUAL(m1.column_end(i)-m1.column_begin(i), m1.size1());
		std::size_t k = 0;
		for(Iter it = m1.column_begin(i); it != m1.column_end(i); ++it,++k){
			BOOST_CHECK_EQUAL(k,it.index());
			*it=0;
			BOOST_CHECK_EQUAL(*it,0);
			BOOST_CHECK_EQUAL(m1(k,i),0);
			*it = m2(k,i);
			BOOST_CHECK_EQUAL(*it,m2(k,i));
			BOOST_CHECK_EQUAL(m1(k,i),m2(k,i));
			*it=0;
			BOOST_CHECK_EQUAL(*it,0);
			BOOST_CHECK_EQUAL(m1(k,i),0);
		}
		//test that the actual iterated length equals the number of elements
		BOOST_CHECK_EQUAL(k, m1.size1());
	}
}
template<class V1, class V2>
void checkDenseVectorEqual(V1 const& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	//indexed access
	for(std::size_t i = 0; i != v2.size(); ++i){
		BOOST_CHECK_EQUAL(v1(i),v2(i));
	}
	//iterator access rows
	typedef typename V1::const_iterator Iter;
	BOOST_REQUIRE_EQUAL(v1.end()-v1.begin(), v1.size());
	std::size_t k = 0;
	for(Iter it = v1.begin(); it != v1.end(); ++it,++k){
		BOOST_CHECK_EQUAL(k,it.index());
		BOOST_CHECK_EQUAL(*it,v2(k));
	}
	//test that the actual iterated length equals the number of elements
	BOOST_CHECK_EQUAL(k, v2.size());
}

template<class V1, class V2>
void checkDenseVectorAssignment(V1& v1, V2 const& v2){
	BOOST_REQUIRE_EQUAL(v1.size(),v2.size());
	//indexed access
	for(std::size_t i = 0; i != v2.size(); ++i){
		v1(i) = 0;
		BOOST_CHECK_EQUAL(v1(i),0);
		v1(i) = v2(i);
		BOOST_CHECK_EQUAL(v1(i),v2(i));
		v1(i) = 0;
		BOOST_CHECK_EQUAL(v1(i),0);
	}
	//iterator access rows
	typedef typename V1::iterator Iter;
	BOOST_REQUIRE_EQUAL(v1.end()-v1.begin(), v1.size());
	std::size_t k = 0;
	for(Iter it = v1.begin(); it != v1.end(); ++it,++k){
		BOOST_CHECK_EQUAL(k,it.index());
		*it = 0;
		BOOST_CHECK_EQUAL(v1(k),0);
		*it = v2(k);
		BOOST_CHECK_EQUAL(v1(k),v2(k));
		*it = 0;
		BOOST_CHECK_EQUAL(v1(k),0);
	}
	//test that the actual iterated length equals the number of elements
	BOOST_CHECK_EQUAL(k, v2.size());
}

std::size_t Dimensions1 = 4;
std::size_t Dimensions2 = 5;
struct MatrixProxyFixture
{
	blas::matrix<double,blas::row_major> denseData;
	blas::matrix<double,blas::column_major> denseDataColMajor;
	
	MatrixProxyFixture():denseData(Dimensions1,Dimensions2),denseDataColMajor(Dimensions1,Dimensions2){
		for(std::size_t row=0;row!= Dimensions1;++row){
			for(std::size_t col=0;col!=Dimensions2;++col){
				denseData(row,col) = row*Dimensions2+col+5;
				denseDataColMajor(row,col) = row*Dimensions2+col+5;
			}
		}
	}
};

BOOST_FIXTURE_TEST_SUITE(data, MatrixProxyFixture);

BOOST_AUTO_TEST_CASE( LinAlg_Dense_Subrange ){
	//all possible combinations of ranges on the data matrix
	for(std::size_t rowEnd=0;rowEnd!= Dimensions1;++rowEnd){
		for(std::size_t rowBegin =0;rowBegin <= rowEnd;++rowBegin){//<= for 0 range
			for(std::size_t colEnd=0;colEnd!=Dimensions2;++colEnd){
				for(std::size_t colBegin=0;colBegin != colEnd;++colBegin){
					std::size_t size1= rowEnd-rowBegin;
					std::size_t size2= colEnd-colBegin;
					blas::matrix<double> mTest(size1,size2);
					for(std::size_t i = 0; i != size1; ++i){
						for(std::size_t j = 0; j != size2; ++j){
							mTest(i,j) = denseData(i+rowBegin,j+colBegin);
						}
					}
					checkDenseMatrixEqual(
						subrange(denseData,rowBegin,rowEnd,colBegin,colEnd),
						mTest
					);
					blas::matrix<double> newData(Dimensions1,Dimensions2,0);
					blas::matrix<double,blas::column_major> newDataColMaj(Dimensions1,Dimensions2,0);
					blas::matrix_range<blas::matrix<double> > rangeTest
					= subrange(newData,rowBegin,rowEnd,colBegin,colEnd);
					blas::matrix_range<blas::matrix<double,blas::column_major> > rangeTestColMaj
					= subrange(newDataColMaj,rowBegin,rowEnd,colBegin,colEnd);
					checkDenseMatrixAssignment(rangeTest,mTest);
					checkDenseMatrixAssignment(rangeTestColMaj,mTest);
					
					//check assignment
					{
						rangeTest=mTest;
						blas::matrix<double> newData2(Dimensions1,Dimensions2,0);
						blas::matrix_range<blas::matrix<double> > rangeTest2
						= subrange(newData2,rowBegin,rowEnd,colBegin,colEnd);
						rangeTest2=rangeTest;
						for(std::size_t i = 0; i != size1; ++i){
							for(std::size_t j = 0; j != size2; ++j){
								BOOST_CHECK_EQUAL(newData(i+rowBegin,j+colBegin),mTest(i,j));
								BOOST_CHECK_EQUAL(newData2(i+rowBegin,j+colBegin),mTest(i,j));
							}
						}
					}
					
					//check clear
					for(std::size_t i = 0; i != size1; ++i){
						for(std::size_t j = 0; j != size2; ++j){
							rangeTest(i,j) = denseData(i+rowBegin,j+colBegin);
							rangeTestColMaj(i,j) = denseData(i+rowBegin,j+colBegin);
						}
					}
					rangeTest.clear();
					rangeTestColMaj.clear();
					for(std::size_t i = 0; i != size1; ++i){
						for(std::size_t j = 0; j != size2; ++j){
							BOOST_CHECK_EQUAL(rangeTest(i,j),0);
							BOOST_CHECK_EQUAL(rangeTestColMaj(i,j),0);
						}
					}
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_Dense_row){
	for(std::size_t r = 0;r != Dimensions1;++r){
		blas::vector<double> vecTest(Dimensions2);
		for(std::size_t i = 0; i != Dimensions2; ++i)
			vecTest(i) = denseData(r,i);
		checkDenseVectorEqual(row(denseData,r),vecTest);
		checkDenseVectorEqual(row(denseDataColMajor,r),vecTest);
		blas::matrix<double> newData(Dimensions1,Dimensions2,0);
		blas::matrix<double,blas::column_major> newDataColMajor(Dimensions1,Dimensions2,0);
		blas::matrix_row<blas::matrix<double> > rowTest = row(newData,r);
		blas::matrix_row<blas::matrix<double,blas::column_major> > rowTestColMajor = row(newDataColMajor,r);
		checkDenseVectorAssignment(rowTest,vecTest);
		checkDenseVectorAssignment(rowTestColMajor,vecTest);
		
		//check assignment
		{
			rowTest=vecTest;
			blas::matrix<double> newData2(Dimensions1,Dimensions2,0);
			blas::matrix_row<blas::matrix<double> > rowTest2 = row(newData2,r);
			rowTest2=rowTest;
			for(std::size_t i = 0; i != Dimensions2; ++i){
				BOOST_CHECK_EQUAL(newData(r,i),vecTest(i));
				BOOST_CHECK_EQUAL(newData2(r,i),vecTest(i));
			}
		}
		//check clear
		for(std::size_t i = 0; i != Dimensions2; ++i){
			rowTest(i) = i;
			rowTestColMajor(i) = i;
		}
		rowTest.clear();
		rowTestColMajor.clear();
		for(std::size_t i = 0; i != Dimensions2; ++i){
			BOOST_CHECK_EQUAL(rowTest(i),0);
			BOOST_CHECK_EQUAL(rowTestColMajor(i),0);
		}
		
	}
}
BOOST_AUTO_TEST_CASE( LinAlg_Dense_column){
	for(std::size_t c = 0;c != Dimensions2;++c){
		blas::vector<double> vecTest(Dimensions1);
		for(std::size_t i = 0; i != Dimensions1; ++i)
			vecTest(i) = denseData(i,c);
		checkDenseVectorEqual(column(denseData,c),vecTest);
		blas::matrix<double> newData(Dimensions1,Dimensions2,0);
		blas::matrix<double,blas::column_major> newDataColMajor(Dimensions1,Dimensions2,0);
		blas::matrix_column<blas::matrix<double> > columnTest = column(newData,c);
		blas::matrix_column<blas::matrix<double,blas::column_major> > columnTestColMajor = column(newDataColMajor,c);
		checkDenseVectorAssignment(columnTest,vecTest);
		checkDenseVectorAssignment(columnTestColMajor,vecTest);
		
		{
			columnTest=vecTest;
			blas::matrix<double> newData2(Dimensions1,Dimensions2,0);
			blas::matrix_column<blas::matrix<double> > columnTest2 = column(newData2,c);
			columnTest2=columnTest;
			for(std::size_t i = 0; i != Dimensions1; ++i){
				BOOST_CHECK_EQUAL(newData(i,c),vecTest(i));
				BOOST_CHECK_EQUAL(newData2(i,c),vecTest(i));
			}
		}
		//check clear
		for(std::size_t i = 0; i != Dimensions1; ++i){
			columnTest(i) = i;
			columnTestColMajor(i) = i;
		}
		columnTest.clear();
		columnTestColMajor.clear();
		for(std::size_t i = 0; i != Dimensions1; ++i){
			BOOST_CHECK_EQUAL(columnTest(i),0);
			BOOST_CHECK_EQUAL(columnTestColMajor(i),0);
		}
	}
}

BOOST_AUTO_TEST_SUITE_END();