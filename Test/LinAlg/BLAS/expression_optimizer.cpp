#define BOOST_TEST_MODULE BLAS_expression_optimizer
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/LinAlg/BLAS/blas.h>

using namespace shark::blas;

BOOST_AUTO_TEST_SUITE (BLAS_Expression_Optimizer)

BOOST_AUTO_TEST_CASE( BLAS_prod_matrix_vector_expression_optimize ){
	// this is a compile-time test to check whether the expressions are actually correctly simplified
	// and return the right types.
	typedef matrix<double,row_major> M1;
	typedef matrix<double,column_major> M2;
	typedef vector<double> V;
	typedef vector<float> V1;
	typedef vector<int> V2;
	
	//we do not have to check transposes as this is automatically done via the checks of prod(vector,matrix)
	//simple cases(identity)
	{
		M1 m1(5,10);
		V v(10);
		matrix_vector_prod<M1,V> e = prod(m1,v);
		(void)e;
	}
	{
		M1 m1(10,5);
		V v(10);
		matrix_vector_prod<matrix_transpose<M1 const>,V> e = prod(v,m1);
		(void)e;
	}
	//matrix-addition
	{
		M1 m1(5,10);
		M2 m2(5,10);
		V v(10);
		vector_addition<matrix_vector_prod<M1,V>,matrix_vector_prod<M2,V> > e = prod(m1+m2,v);
		(void)e;
	}
	{
		M1 m1(5,10);
		M2 m2(5,10);
		V v(10);
		vector_addition<
			matrix_vector_prod<matrix_transpose<M1 const>,V>,
			matrix_vector_prod<matrix_transpose<M2 const>,V> 
		> e = prod(v,m1+m2);
		(void)e;
	}
	//outer product
	{
		V1 v1(5);
		V2 v2(10);
		V v(10);
		vector_scalar_multiply<V1> e = prod(outer_prod(v1,v2),v);
		(void)e;
	}
	{
		V1 v1(5);
		V2 v2(10);
		V v(10);
		vector_scalar_multiply<V1> e = prod(v,outer_prod(v2,v1));
		(void)e;
	}
	//nested product
	{
		M1 m1(5,10);
		M2 m2(10,8);
		V v(8);
		matrix_vector_prod<M1,matrix_vector_prod<M2,V> > e = prod(prod(m1,m2),v);
		(void)e;
	}
	{
		M1 m1(10,5);
		M2 m2(8,10);
		V v(8);
		matrix_vector_prod<
			matrix_transpose<M1 const>,
			matrix_vector_prod<matrix_transpose<M2 const>,V>
		> e = prod(v,prod(m2,m1));
		(void)e;
	}
}

BOOST_AUTO_TEST_CASE( BLAS_prod_matrix_row_optimize ){
	// this is a compile-time test to check whether the expressions are actually correctly simplified
	// and return the right types.
	typedef matrix<double,row_major> M1;
	typedef matrix<double,column_major> M2;
	typedef vector<double> V;
	typedef vector<float> V1;
	typedef vector<int> V2;
	
	//simple proxy cases
	{
		M1 m1(5,10);
		temporary_proxy<matrix_row<M1> > e1 = row(m1,1);
		matrix_row<M1 const> e2 = row(static_cast<M1 const&>(m1),1);
		temporary_proxy<matrix_row<matrix_transpose<M1> > > e3 = column(m1,1);
		matrix_row<matrix_transpose<M1 const> > e4 = column(static_cast<M1 const&>(m1),1);
	}
	//matrix sum
	{
		M1 m1(5,10);
		M2 m2(5,10);
		vector_addition<matrix_row<M1 const>,matrix_row<M2 const> > e1 = row(m1+m2,1);
		vector_addition<
			matrix_row<matrix_transpose<M1 const> >,
			matrix_row<matrix_transpose<M2 const> >
		> e2 = column(m1+m2,1);
	}
	
	//scaled matrix
	{
		M1 m1(5,10);
		double alpha = 2;
		vector_scalar_multiply<matrix_row<M1 const> > e1 = row(alpha*m1,1);
		vector_scalar_multiply<matrix_row<matrix_transpose<M1 const> > > e2 = column(alpha*m1,1);
	}
	
	//matrix unary
	{
		M1 m1(5,10);
		typedef scalar_sqr<double> F;
		vector_unary<matrix_row<M1 const>, F> e1 = row(sqr(m1),1);
		vector_unary<matrix_row<matrix_transpose<M1 const> >,F> e2 = column(sqr(m1),1);
	}
	//matrix binary
	{
		M1 m1(5,10);
		M2 m2(5,10);
		typedef scalar_binary_multiply<double,double> F;
		vector_binary<matrix_row<M1 const>,matrix_row<M2 const>,F> e1 = row(m1*m2,1);
		vector_binary<matrix_row<matrix_transpose<M1 const>>,matrix_row<matrix_transpose<M2 const> >,F> e2 = column(m1*m2,1);
	}
	
	//matrix prod
	{
		M1 m1(5,10);
		M2 m2(10,5);
		matrix_vector_prod<matrix_transpose<M2 const>,matrix_row<M1 const> > e1 = row(prod(m1,m2),1);
		matrix_vector_prod<matrix_reference<M1 const>, matrix_row<matrix_transpose<M2 const> >> e2 = column(prod(m1,m2),1);
	}
}

BOOST_AUTO_TEST_SUITE_END()
