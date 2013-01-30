
#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE ML_PrecomputedMatrix
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <shark/Data/PrecomputedMatrix.h>
#include <shark/Data/Libsvm.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>
#include <shark/Models/Kernels/LinearKernel.h>

#include <iostream>
#include <sstream>

using namespace shark;

//first, use the example provided in the libsvm readme file
const char test_classification[] = "-1  1:1 2:1 3:1 4:1.0000 \n\
         1      2:3     4:3\n\
  +1          3:1e0     \n\
-1 1:10e-1 3:1\n";
const char test_mc_classification[] = "1  1:1 2:1 3:1 4:1.0000 \n\
         3      2:3     4:3\n\
  2          3:1e0     \n\
1 1:10e-1 3:1\n";
const char test_regression[] = "2.2  1:1 2:1 3:1 4:1.0000 \n\
0.07      2:3     4:3\n\
	-12          3:1e0     \n\
 +1.4e-1 1:10e-1 3:1\n";

BOOST_AUTO_TEST_CASE( Set_PrecomputedMatrix )
{
	
	
	// load data and set up vars
	std::stringstream ssc(test_classification), ssmcc(test_mc_classification); //dense class., dense mc-class.
	std::stringstream ssr(test_regression), sssc(test_classification); // dense regr., sparse classif.
	std::vector<RealVector> xc, xmcc, xr;
	std::vector<unsigned int> yc, ysc, ymcc;
	std::vector<double> yr;
	
	detail::import_libsvm(xc, yc, ssc); //dense classif.
	LabeledData<RealVector, unsigned int> test_ds_c(xc, yc);
	detail::import_libsvm(xmcc, ymcc, ssmcc); //dense mc-classif.
	LabeledData<RealVector, unsigned int> test_ds_mcc(xmcc, ymcc);
	detail::import_libsvm(xr, yr, ssr); //dense regression
	LabeledData<RealVector, double> test_ds_r(xr, yr);
	// also test one sparse format:
	std::vector<CompressedRealVector> xsc;
	detail::import_libsvm(xsc, ysc, sssc); //sparse classif.
	LabeledData<CompressedRealVector, unsigned int> test_ds_sc(xsc, ysc);
	
	DenseLinearKernel kernel;
	CompressedLinearKernel skernel; //sparse
	
	// no scaling
	export_kernel_matrix( test_ds_c, kernel, "test_output/check_kernelmatrix_c_none.libsvm", NONE, false, 15 );
	export_kernel_matrix( test_ds_mcc, kernel, "test_output/check_kernelmatrix_mcc_none.libsvm", NONE, false, 15 );
	export_kernel_matrix( test_ds_r, kernel, "test_output/check_kernelmatrix_r_none.libsvm", NONE, false, 15 );
	// trace=1
	export_kernel_matrix( test_ds_c, kernel, "test_output/check_kernelmatrix_c_trace.libsvm", MULTIPLICATIVE_TRACE_ONE, false, 15 );
	export_kernel_matrix( test_ds_mcc, kernel, "test_output/check_kernelmatrix_mcc_trace.libsvm", MULTIPLICATIVE_TRACE_ONE, false, 15 );
	export_kernel_matrix( test_ds_r, kernel, "test_output/check_kernelmatrix_r_trace.libsvm", MULTIPLICATIVE_TRACE_ONE, false, 15 );
	// trace=N
	export_kernel_matrix( test_ds_c, kernel, "test_output/check_kernelmatrix_c_traceN.libsvm", MULTIPLICATIVE_TRACE_N, false, 15 );
	export_kernel_matrix( test_ds_mcc, kernel, "test_output/check_kernelmatrix_mcc_traceN.libsvm", MULTIPLICATIVE_TRACE_N, false, 15 );
	export_kernel_matrix( test_ds_r, kernel, "test_output/check_kernelmatrix_r_traceN.libsvm", MULTIPLICATIVE_TRACE_N, false, 15 );
	// var=1
	export_kernel_matrix( test_ds_c, kernel, "test_output/check_kernelmatrix_c_var.libsvm", MULTIPLICATIVE_VARIANCE_ONE, false, 15 );
	export_kernel_matrix( test_ds_mcc, kernel, "test_output/check_kernelmatrix_mcc_var.libsvm", MULTIPLICATIVE_VARIANCE_ONE, false, 15 );
	export_kernel_matrix( test_ds_r, kernel, "test_output/check_kernelmatrix_r_var.libsvm", MULTIPLICATIVE_VARIANCE_ONE, false, 15 );
	// center
	export_kernel_matrix( test_ds_c, kernel, "test_output/check_kernelmatrix_c_center.libsvm", CENTER_ONLY, false, 15 );
	export_kernel_matrix( test_ds_mcc, kernel, "test_output/check_kernelmatrix_mcc_center.libsvm", CENTER_ONLY, false, 15 );
	export_kernel_matrix( test_ds_r, kernel, "test_output/check_kernelmatrix_r_center.libsvm", CENTER_ONLY, false, 15 );
	// center and tr=1
	export_kernel_matrix( test_ds_c, kernel, "test_output/check_kernelmatrix_c_center_tr.libsvm", CENTER_AND_MULTIPLICATIVE_TRACE_ONE, false, 15 );
	export_kernel_matrix( test_ds_mcc, kernel, "test_output/check_kernelmatrix_mcc_center_tr.libsvm", CENTER_AND_MULTIPLICATIVE_TRACE_ONE, false, 15 );
	export_kernel_matrix( test_ds_r, kernel, "test_output/check_kernelmatrix_r_center_tr.libsvm", CENTER_AND_MULTIPLICATIVE_TRACE_ONE, false, 15 );
	// sparse classif., no scaling
	export_kernel_matrix( test_ds_sc, skernel, "test_output/check_kernelmatrix_sparse_c_none.libsvm", NONE, false, 15 );
	
	// mt: todo: add tests here once read-in is supported
	
}
