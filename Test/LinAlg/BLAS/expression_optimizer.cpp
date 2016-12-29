#define BOOST_TEST_MODULE BLAS_expression_optimizer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Core/Shark.h>
#include <shark/LinAlg/BLAS/blas.h>
#include <random>

std::default_random_engine e1(42);

using namespace shark::blas;

struct ExpressionOptimizerFixture{
	ExpressionOptimizerFixture():e1(42){}
	std::default_random_engine e1;
	std::normal_distribution<> dist;
		
	matrix<double> create_matrix(std::size_t rows, std::size_t columns){
		matrix<double> mat(rows,columns);
		for(std::size_t i = 0; i != rows; ++i){
			for(std::size_t j = 0; j != columns; ++j){
				mat(i,j) = dist(e1);
			}
		}
		return mat;
	}

	vector<double> create_vector(std::size_t rows){
		vector<double> vec(rows);
		for(std::size_t i = 0; i != rows; ++i){
			vec(i) = dist(e1);
		}
		return vec;
	}
};

BOOST_FIXTURE_TEST_SUITE(BLAS_Expression_Optimizer,ExpressionOptimizerFixture)

BOOST_AUTO_TEST_CASE( BLAS_prod_matrix_vector_expression_optimize ){
	typedef matrix<double,row_major> M1;
	typedef matrix<double,column_major> M2;
	typedef vector<double> V;
	typedef vector<float> V1;
	typedef vector<double> V2;
	
	//we do not have to check transposes as this is automatically done via the checks of prod(vector,matrix)
	//simple cases(identity)
	{
		M1 m1 = create_matrix(5,10);
		V v = create_vector(10);
		matrix_vector_prod<M1,V> e = prod(m1,v);
		BOOST_CHECK_SMALL(norm_inf(m1 - e.matrix()), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(v - e.vector()), 1.e-10);
	}
	{
		M1 m1 = create_matrix(5,10);
		V v = create_vector(5);
		matrix_vector_prod<matrix_transpose<M1 const>,V> e = prod(v,m1);
		BOOST_CHECK_SMALL(norm_inf(trans(m1) - e.matrix()), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(v - e.vector()), 1.e-10);
	}
	//matrix-addition
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(5,10);
		V v = create_vector(10);
		vector_addition<matrix_vector_prod<M1,V>,matrix_vector_prod<M2,V> > e = prod(m1+m2,v);
		BOOST_CHECK_SMALL(norm_inf(e - prod(matrix<double>(m1+m2),v)), 1.e-10);
	}
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(5,10);
		V v = create_vector(5);
		vector_addition<
			matrix_vector_prod<matrix_transpose<M1 const>,V>,
			matrix_vector_prod<matrix_transpose<M2 const>,V> 
		> e = prod(v,m1 + m2);
		BOOST_CHECK_SMALL(norm_inf(e - prod(v,matrix<double>(m1+m2))), 1.e-10);
	}
	//outer product
	{
		V2 v1 = create_vector(5);
		V1 v2 = create_vector(10);
		V v = create_vector(10);
		vector_scalar_multiply<V2> e = prod(outer_prod(v1,v2),v);
		BOOST_CHECK_SMALL(norm_inf(e - prod(matrix<double>(outer_prod(v1,v2)),v)), 1.e-10);
	}
	{
		V1 v1 = create_vector(5);
		V2 v2 = create_vector(10);
		V v = create_vector(5);
		vector_scalar_multiply<V2> e = prod(v,outer_prod(v1,v2));
		M1 temp = outer_prod(v1,v2);
		BOOST_CHECK_SMALL(norm_inf(e - prod(v,temp)), 1.e-10);
	}
	//nested product
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(10,8);
		V v = create_vector(8);
		matrix_vector_prod<M1,matrix_vector_prod<M2,V> > e = prod(prod(m1,m2),v);
		BOOST_CHECK_SMALL(norm_inf(e - prod(matrix<double>(prod(m1,m2)),v)), 1.e-10);
	}
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(10,8);
		V v = create_vector(5);
		matrix_vector_prod<
			matrix_transpose<M2 const>,
			matrix_vector_prod<matrix_transpose<M1 const>,V>
		> e = prod(v,prod(m1,m2));
		BOOST_CHECK_SMALL(norm_inf(e - prod(v,matrix<double>(prod(m1,m2)))), 1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( BLAS_prod_matrix_row_optimize ){
	typedef matrix<double,row_major> M1;
	typedef matrix<double,column_major> M2;
	
	//simple proxy cases
	{
		M1 m1 = create_matrix(5,10);
		temporary_proxy<matrix_row<M1> >&& e1 = row(m1,1);
		matrix_row<M1 const> e2 = row(static_cast<M1 const&>(m1),1);
		temporary_proxy<matrix_row<matrix_transpose<M1> > >&& e3 = column(m1,1);
		matrix_row<matrix_transpose<M1 const> > e4 = column(static_cast<M1 const&>(m1),1);
		BOOST_CHECK_SMALL(norm_inf(e1 - e2), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(e3 - e4), 1.e-10);
	}
	//matrix sum
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(5,10);
		vector_addition<matrix_row<M1 const>,matrix_row<M2 const> > e1 = row(m1+m2,1);
		vector_addition<
			matrix_row<matrix_transpose<M1 const> >,
			matrix_row<matrix_transpose<M2 const> >
		> e2 = column(m1+m2,1);
		BOOST_CHECK_SMALL(norm_inf(e1 - row(matrix<double>(m1+m2),1)), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(e2 - column(matrix<double>(m1+m2),1)), 1.e-10);
	}
	
	//scaled matrix
	{
		M1 m1 = create_matrix(5,10);
		double alpha = 2;
		vector_scalar_multiply<matrix_row<M1 const> > e1 = row(alpha*m1,1);
		vector_scalar_multiply<matrix_row<matrix_transpose<M1 const> > > e2 = column(alpha*m1,1);
		BOOST_CHECK_SMALL(norm_inf(e1 - row(matrix<double>(alpha*m1),1)), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(e2 - column(matrix<double>(alpha*m1),1)), 1.e-10);
	}
	
	//matrix unary
	{
		M1 m1 = create_matrix(5,10);
		typedef functors::scalar_sqr<double> F;
		vector_unary<matrix_row<M1 const>, F> e1 = row(sqr(m1),1);
		vector_unary<matrix_row<matrix_transpose<M1 const> >,F> e2 = column(sqr(m1),1);
		BOOST_CHECK_SMALL(norm_inf(e1 - row(matrix<double>(sqr(m1)),1)), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(e2 - column(matrix<double>(sqr(m1)),1)), 1.e-10);
	}
	//matrix binary
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(5,10);
		typedef functors::scalar_binary_multiply<double> F;
		vector_binary<matrix_row<M1 const>,matrix_row<M2 const>,F> e1 = row(m1*m2,1);
		vector_binary<matrix_row<matrix_transpose<M1 const>>,matrix_row<matrix_transpose<M2 const> >,F> e2 = column(m1*m2,1);
		BOOST_CHECK_SMALL(norm_inf(e1 - row(matrix<double>(m1*m2),1)), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(e2 - column(matrix<double>(m1*m2),1)), 1.e-10);
	}
	
	//matrix prod
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(10,8);
		matrix_vector_prod<matrix_transpose<M2 const>,matrix_row<M1 const> > e1 = row(prod(m1,m2),1);
		matrix_vector_prod<matrix_reference<M1 const>, matrix_row<matrix_transpose<M2 const> >> e2 = column(prod(m1,m2),1);
		BOOST_CHECK_SMALL(norm_inf(e1 - row(matrix<double>(prod(m1,m2)),1)), 1.e-10);
		BOOST_CHECK_SMALL(norm_inf(e2 - column(matrix<double>(prod(m1,m2)),1)), 1.e-10);
	}
}


BOOST_AUTO_TEST_CASE( BLAS_prod_vector_range_optimize ){
	typedef matrix<double,row_major> M;
	typedef vector<double> V1;
	typedef vector<float> V2;
	
	//simple proxy cases
	{
		V1 v1 = create_vector(10);
		temporary_proxy<vector_range<V1> >&& e1 = subrange(v1,1,4);
		vector_range<V1 const> e2 = subrange(static_cast<V1 const&>(v1),1,4);
		BOOST_CHECK_SMALL(norm_inf(e1 - e2), 1.e-10);
	}
	//vector sum
	{
		V1 v1 = create_vector(10);
		V2 v2 = create_vector(10);
		vector_addition<vector_range<V1 const>,vector_range<V2 const> > e1 = subrange(v1+v2,1,4);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(vector<double>(v1+v2),1,4)), 1.e-10);
	}
	
	//scaled vector
	{
		V1 v1 = create_vector(10);
		double alpha = 2;
		vector_scalar_multiply<vector_range<V1 const> > e1 = subrange(alpha*v1,1,4);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(vector<double>(alpha*v1),1,4)), 1.e-10);
	}
	
	//vector unary
	{
		V1 v1 = create_vector(10);
		typedef functors::scalar_sqr<double> F;
		vector_unary<vector_range<V1 const>, F> e1 = subrange(sqr(v1),1,4);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(vector<double>(sqr(v1)),1,4)), 1.e-10);
	}
	//vector binary
	{
		V1 v1 = create_vector(10);
		V2 v2 = create_vector(10);
		typedef functors::scalar_binary_multiply<double> F;
		vector_binary<vector_range<V1 const>,vector_range<V2 const>,F> e1 = subrange(v1*v2,1,4);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(vector<double>(v1*v2),1,4)), 1.e-10);
	}
	
	//vector prod
	{
		V1 v1 = create_vector(10);
		M m1 = create_matrix(7,10);
		matrix_vector_prod<matrix_range<M const>,V1 const> e1 = subrange(prod(m1,v1),1,4);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(vector<double>(prod(m1,v1)),1,4)), 1.e-10);
	}
}

BOOST_AUTO_TEST_CASE( BLAS_prod_matrix_range_optimize ){
	typedef matrix<double,row_major> M1;
	typedef matrix<double,column_major> M2;
	
	//simple proxy cases
	{
		M1 m1 = create_matrix(5,10);
		temporary_proxy<matrix_range<M1> >&& e1 = subrange(m1,1,4,3,7);
		matrix_range<M1 const> e2 = subrange(static_cast<M1 const&>(m1),1,4,3,7);
		BOOST_CHECK_SMALL(norm_inf(e1 - e2), 1.e-10);
	}
	//matrix sum
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(5,10);
		matrix_addition<matrix_range<M1 const>,matrix_range<M2 const> > e1 = subrange(m1+m2,1,4,3,7);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(matrix<double>(m1+m2),1,4,3,7)), 1.e-10);
	}
	
	//scaled matrix
	{
		M1 m1 = create_matrix(5,10);
		double alpha = 2;
		matrix_scalar_multiply<matrix_range<M1 const> > e1 = subrange(alpha*m1,1,4,3,7);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(matrix<double>(alpha*m1),1,4,3,7)), 1.e-10);
	}
	
	//matrix unary
	{
		M1 m1 = create_matrix(5,10);
		typedef functors::scalar_sqr<double> F;
		matrix_unary<matrix_range<M1 const>, F> e1 = subrange(sqr(m1),1,4,3,7);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(matrix<double>(sqr(m1)),1,4,3,7)), 1.e-10);
	}
	//matrix binary
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(5,10);
		typedef functors::scalar_binary_multiply<double> F;
		matrix_binary<matrix_range<M1 const>,matrix_range<M2 const>,F> e1 = subrange(m1*m2,1,4,3,7);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(matrix<double>(m1*m2),1,4,3,7)), 1.e-10);
	}
	
	//matrix prod
	{
		M1 m1 = create_matrix(5,10);
		M2 m2 = create_matrix(10,12);
		matrix_matrix_prod<matrix_range<M1 const>,matrix_range<M2 const> > e1 = subrange(prod(m1,m2),1,4,3,7);
		BOOST_CHECK_SMALL(norm_inf(e1 - subrange(matrix<double>(prod(m1,m2)),1,4,3,7)), 1.e-10);
	}
}

BOOST_AUTO_TEST_SUITE_END()
