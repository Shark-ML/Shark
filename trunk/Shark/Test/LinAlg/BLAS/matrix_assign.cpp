#define BOOST_TEST_MODULE LinAlg_BLAS_VectorAssign
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark;

template<class M1, class M2>
void checkMatrixEqual(M1 const& m1, M2 const& m2){
	BOOST_REQUIRE_EQUAL(m1.size1(),m2.size1());
	BOOST_REQUIRE_EQUAL(m1.size2(),m2.size2());
	for(std::size_t i = 0; i != m2.size1(); ++i){
		for(std::size_t j = 0; j != m2.size2(); ++j){
			BOOST_CHECK_EQUAL(m1(i,j),m2(i,j));
		}
	}
}



//////////////////////////////////////////////////////
//////SIMPLE ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_SUITE (LinAlg_BLAS_matrix_assign)

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Dense_Matrix_Assign ){
	std::cout<<"testing direct dense-dense assignment"<<std::endl;
	blas::matrix<unsigned int,blas::row_major> source_row_major(10,20);
	blas::matrix<unsigned int,blas::column_major> source_column_major(10,20);

	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j)+2;
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Sparse_Matrix_Assign ){
	std::cout<<"\ntesting direct dense-sparse assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20,0);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 1; j < 20; j+=(i+1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  2*(20*i+1)+1;//source_row_major(i,j)+2;
		}
	}
	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target(10,20);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 0; j != 20; ++j){
				target(i,j) = 3*(20*i+1)+2;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Dense_Matrix_Assign ){
	std::cout<<"\ntesting direct sparse-dense assignment"<<std::endl;
	blas::matrix<unsigned int,blas::row_major> source_row_major(10,20);
	blas::matrix<unsigned int,blas::column_major> source_column_major(10,20);

	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j)+2;
		}
	}
	
	//test all 4 combinations of row/column major
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 1; j < 20; j+=(i+1)){
				target(i,j) = 4*(20*i+1)+9;
			}
		}
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 1; j < 20; j+=(i+1)){
				target(i,j) =  4*(20*i+1)+9;
			}
		}
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 1; j < 20; j+=(i+1)){
				target(i,j) = 4*(20*i+1)+9;
			}
		}
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 1; j < 20; j+=(i+1)){
				target(i,j) =  4*(20*i+1)+9;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Sparse_Matrix_Assign ){
	std::cout<<"\ntesting direct sparse-sparse assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20,0);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 1; j < 20; j+=(i+1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j);
		}
	}
	
	//test all 4 combinations of row/column major
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ for(std::size_t j = 1; j < 20; j+=(i+1)){
				//~ target(i,j) = 4*(20*i+1)+9;
			//~ }
		//~ }
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target(10,20,0);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ for(std::size_t j = 1; j < 20; j+=(i+1)){
				//~ target(i,j) = 4*(20*i+1)+9;
			//~ }
		//~ }
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		//~ for(std::size_t i = 0; i != 10; ++i){
			//~ for(std::size_t j = 1; j < 20; j+=(i+1)){
				//~ target(i,j) = 4*(20*i+1)+9;
			//~ }
		//~ }
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign(target,source_row_major);
		checkMatrixEqual(target,source_row_major);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		for(std::size_t i = 0; i != 10; ++i){
			for(std::size_t j = 1; j < 20; j+=(i+1)){
				target(i,j) =  4*(20*i+1)+9;
			}
		}
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign(target,source_column_major);
		checkMatrixEqual(target,source_column_major);
	}
}



//////////////////////////////////////////////////////
//////PLUS ASSIGNMENT
//////////////////////////////////////////////////////

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Dense_Matrix_Plus_Assign ){
	std::cout<<"\ntesting dense-dense functor assignment"<<std::endl;
	blas::matrix<unsigned int,blas::row_major> source_row_major(10,20);
	blas::matrix<unsigned int,blas::column_major> source_column_major(10,20);
	blas::matrix<unsigned int,blas::row_major> preinit(10,20);
	blas::matrix<unsigned int,blas::row_major> result(10,20);

	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) =  source_row_major(i,j);
			preinit(i,j) = 3*(20*i+1)+2;
			result(i,j) = preinit(i,j)+source_row_major(i,j);
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
}

BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Dense_Sparse_Matrix_Plus_Assign ){
	std::cout<<"\ntesting dense-sparse functor assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	blas::matrix<unsigned int,blas::row_major> preinit(10,20);
	blas::matrix<unsigned int,blas::row_major> result(10,20);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 1; j < 20; j+=(i+1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) = 2*(20*i+1)+1;
		}
	}

	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j != 20; ++j){
			preinit(i,j) = 3*(20*i+1)+2;
			result(i,j) = preinit(i,j)+source_row_major(i,j);
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::column_major> target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
}


BOOST_AUTO_TEST_CASE( LinAlg_BLAS_Sparse_Sparse_Matrix_Plus_Assign ){
	std::cout<<"\ntesting sparse-sparse functor assignment"<<std::endl;
	blas::compressed_matrix<unsigned int> source_row_major(10,20);
	blas::compressed_matrix<unsigned int> source_column_major_base(20,10);
	blas::matrix_transpose<blas::compressed_matrix<unsigned int> > source_column_major(source_column_major_base);
	blas::compressed_matrix<unsigned int> preinit(10,20);
	blas::compressed_matrix<unsigned int> result(10,20);
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 1; j < 20; j+=(i+1)){
			source_row_major(i,j) = 2*(20*i+1)+1;
			source_column_major(i,j) = 2*(20*i+1)+1;
		}
	}

	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j < 20; j+=(i+2)/2){
			preinit(i,j) = 3*(20*i+1)+2;
		}
	}
	
	for(std::size_t i = 0; i != 10; ++i){
		for(std::size_t j = 0; j < 20; ++j){
			int r = preinit(i,j)+source_row_major(i,j);
			if(r != 0)
				result(i,j) = r;
		}
	}

	//test all 4 combinations of row/column major
	{
		blas::compressed_matrix<unsigned int> target = preinit;
		std::cout<<"testing row-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::matrix<unsigned int,blas::row_major> target = preinit;
		std::cout<<"testing row-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		target = preinit;
		std::cout<<"testing column-row"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_row_major);
		checkMatrixEqual(target,result);
	}
	
	{
		blas::compressed_matrix<unsigned int> target_base(20,10);
		blas::matrix_transpose<blas::compressed_matrix<unsigned int> > target(target_base);
		target = preinit;
		std::cout<<"testing column-column"<<std::endl;
		blas::kernels::assign<blas::scalar_plus_assign>(target,source_column_major);
		checkMatrixEqual(target,result);
	}
}
BOOST_AUTO_TEST_SUITE_END()
