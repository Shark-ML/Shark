#define BOOST_TEST_MODULE ML_KernelFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include "KernelDerivativeTestHelper.h"

#include <boost/math/constants/constants.hpp>

#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/MonomialKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/ArdKernel.h>
#include <shark/Models/Kernels/WeightedSumKernel.h>
#include <cmath>

using namespace shark;

BOOST_AUTO_TEST_SUITE (Models_Kernels_WeightedSumKernel)

BOOST_AUTO_TEST_CASE( DenseWeightedSumKernel_Test )
{
    const double gamma1 = 0.1;
    const double gamma2 = 0.01;
    RealVector testParams(2);
    testParams(0) = 1;
    testParams(1) = gamma2;

    DenseRbfKernel baseKernel1(gamma1);
    DenseRbfKernel baseKernel2(2*gamma2);
    std::vector<AbstractKernelFunction<RealVector>* > kernels;
    kernels.push_back(&baseKernel1);
    kernels.push_back(&baseKernel2);
    DenseWeightedSumKernel kernel(kernels);
    kernel.setAdaptive(1, true);

    //now test whether the parametervector is created correctly
    kernel.setParameterVector(testParams);
    RealVector parameter=kernel.parameterVector();
    BOOST_CHECK_SMALL(norm_sqr(parameter-testParams), 1.e-15);
    //and check whether all gamma values are correct
    BOOST_CHECK_SMALL(baseKernel1.gamma() - gamma1, 1e-13);
    BOOST_CHECK_SMALL(baseKernel2.gamma() - gamma2, 1e-13);

    //testpoints
    RealVector x1(2);
    x1(0)=2;
    x1(1)=1;
    RealVector x2(2);
    x2(0)=-2;
    x2(1)=1;

    double k1 = baseKernel1.eval(x1,x2);
    double k2 = baseKernel2.eval(x1,x2);
    double numeratorResult = k1 + boost::math::constants::e<double>() * k2;
    double result = numeratorResult / (1 + boost::math::constants::e<double>());

    //testbatches
    RealMatrix batchX1(2,2);
    batchX1(0,0)=2;
    batchX1(0,1)=1;
    batchX1(1,0)=1;
    batchX1(1,1)=3;
    RealMatrix batchX2(2,2);
    batchX2(0,0)=-2;
    batchX2(1,0)=1;
    batchX2(1,0)=3;
    batchX2(1,1)=3;

    boost::shared_ptr<State> stateK1Batch = baseKernel1.createState();
    boost::shared_ptr<State> stateK2Batch = baseKernel2.createState();
    RealMatrix k1Batch,k2Batch;
    baseKernel1.eval(batchX1,batchX2,k1Batch,*stateK1Batch);
    baseKernel2.eval(batchX1,batchX2,k2Batch,*stateK2Batch);
    RealMatrix numeratorResultBatch = k1Batch + boost::math::constants::e<double>() * k2Batch;
    RealMatrix resultBatch = numeratorResultBatch / (1 + boost::math::constants::e<double>());

    //evaluate point
    double test = kernel.eval(x1,x2);
    BOOST_REQUIRE_SMALL(result - test, 1.e-15);

    //evaluate batch
    RealMatrix testBatch,testBatch2;
    kernel.eval(batchX1,batchX2,testBatch);
    boost::shared_ptr<State> stateBatch = kernel.createState();
    kernel.eval(batchX1,batchX2,testBatch2,*stateBatch);
    BOOST_REQUIRE_SMALL(resultBatch(0,0) - testBatch(0,0), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(0,0) - testBatch2(0,0), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(0,1) - testBatch(0,1), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(0,1) - testBatch2(0,1), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(1,0) - testBatch(1,0), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(1,0) - testBatch2(1,0), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(1,1) - testBatch(1,1), 1.e-15);
    BOOST_REQUIRE_SMALL(resultBatch(1,1) - testBatch2(1,1), 1.e-15);

    //test first derivative
    testKernelDerivative(kernel, 2, 1.e-8,1.e-6,1,1);
    testKernelInputDerivative(kernel, 2, 1.e-8);
}

BOOST_AUTO_TEST_CASE( DenseWeightedSumKernel_Test_Detailed )
{
    unsigned int numdim = 3;
    unsigned int numker = 6;
    DenseRbfKernel        basekernel1(0.1);
    DenseRbfKernel        basekernel2(0.4, true);
    DenseLinearKernel     basekernel3;
    DenseMonomialKernel   basekernel4(3);
    DensePolynomialKernel basekernel5(2, 1.0);
    BOOST_CHECK( !basekernel5.hasFirstParameterDerivative() );
    DenseARDKernel        basekernel6(numdim,0.2);

    std::vector< AbstractKernelFunction<RealVector> * > kernels;
    kernels.push_back(&basekernel1);
    kernels.push_back(&basekernel2);
    kernels.push_back(&basekernel3);
    kernels.push_back(&basekernel4);
    kernels.push_back(&basekernel5);
    kernels.push_back(&basekernel6);
    DenseWeightedSumKernel kernel(kernels);

    // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    unsigned int num_bools = 5000;
    unsigned int num_trials = 50;
    std::vector< bool > cur_bools(numker);
    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        for ( unsigned int j=0; j<num_trials; j++ ) {
            RealVector cur_params(kernel.numberOfParameters());
            for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                cur_params(k) = Rng::discrete(1,10);
            kernel.setParameterVector(cur_params);
            BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        }
    }
    kernel.setAdaptiveAll(true);
    // test setting and getting the entire parameter vector
    for ( unsigned int j=0; j<num_trials; j++ ) {
        RealVector cur_params(kernel.numberOfParameters());
        for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            cur_params(k) = Rng::uni(1.0,3.0);
        cur_params(7) = (unsigned int)cur_params(7);
        kernel.setParameterVector(cur_params);
        BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.parameterVector()(0)-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.gamma()-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel5.parameterVector()(1)-cur_params(8) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(9) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(10) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(11) , 1.e-15);
    }

    // test kernel evals on a test point
    RealVector my_params(12);
    my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1; my_params(7) = 3;
    my_params(8) = 0.5; my_params(9) = 0.6; my_params(10) = 2.0; my_params(11) = -1.1;
    kernel.setParameterVector(my_params);

    RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;

    double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    double k4 = k3*k3*k3;
    double k5 = (k3+0.5)*(k3+0.5)*(k3+0.5);
    double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    double k = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    BOOST_CHECK_SMALL( k/d - kernel.eval(test1,test2), 1e-15 );

    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    }
}

BOOST_AUTO_TEST_CASE( DenseWeightedSumKernel_Test_Detailed_Unconstrained )
{
    unsigned int numdim = 3;
    unsigned int numker = 6;
    DenseRbfKernel        basekernel1(0.1);
    DenseRbfKernel        basekernel2(0.4, true);
    DenseLinearKernel     basekernel3;
    DenseMonomialKernel   basekernel4(3);
    DensePolynomialKernel basekernel5(2, 1.0, true, true);
    BOOST_CHECK( !basekernel5.hasFirstParameterDerivative() );
    DenseARDKernel        basekernel6(numdim,0.2);

    std::vector< AbstractKernelFunction<RealVector> * > kernels;
    kernels.push_back(&basekernel1);
    kernels.push_back(&basekernel2);
    kernels.push_back(&basekernel3);
    kernels.push_back(&basekernel4);
    kernels.push_back(&basekernel5);
    kernels.push_back(&basekernel6);
    DenseWeightedSumKernel kernel(kernels);

    // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    unsigned int num_bools = 5000;
    unsigned int num_trials = 50;
    std::vector< bool > cur_bools(numker);
    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        for ( unsigned int j=0; j<num_trials; j++ ) {
            RealVector cur_params(kernel.numberOfParameters());
            for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                cur_params(k) = Rng::discrete(1,10);
            kernel.setParameterVector(cur_params);
            BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        }
    }
    kernel.setAdaptiveAll(true);
    // test setting and getting the entire parameter vector
    for ( unsigned int j=0; j<num_trials; j++ ) {
        RealVector cur_params(kernel.numberOfParameters());
        for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            cur_params(k) = Rng::uni(1.0,3.0);
        cur_params(7) = (unsigned int)cur_params(7);
        kernel.setParameterVector(cur_params);
        BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.parameterVector()(0)-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.gamma()-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel5.parameterVector()(1)-cur_params(8) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(9) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(10) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(11) , 1.e-15);
    }

    // test kernel evals on a test point
    RealVector my_params(12);
    my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1; my_params(7) = 3;
    my_params(8) = -0.5; my_params(9) = 0.6; my_params(10) = 2.0; my_params(11) = -1.1;
    kernel.setParameterVector(my_params);

    RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;

    double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    double k4 = k3*k3*k3;
    double k5 = (k3+std::exp(-0.5))*(k3+std::exp(-0.5))*(k3+std::exp(-0.5));
    double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    double k = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    BOOST_CHECK_SMALL( k/d - kernel.eval(test1,test2), 1e-15 );

    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    }
}

BOOST_AUTO_TEST_CASE( DenseWeightedSumKernel_Test_Detailed_NoDegreeParam )
{
    unsigned int numdim = 3;
    unsigned int numker = 6;
    DenseRbfKernel        basekernel1(0.1);
    DenseRbfKernel        basekernel2(0.4, true);
    DenseLinearKernel     basekernel3;
    DenseMonomialKernel   basekernel4(3);
    DensePolynomialKernel basekernel5(2, 1.0, false);
    BOOST_CHECK( basekernel5.hasFirstParameterDerivative() );
    basekernel5.setDegree( 3 );
//  basekernel5.setDegree( 0 ); //should fail
    BOOST_CHECK_EQUAL( basekernel5.degree(), 3 );
    basekernel5.setDegree( 2 );
    DenseARDKernel        basekernel6(numdim,0.2);

    std::vector< AbstractKernelFunction<RealVector> * > kernels;
    kernels.push_back(&basekernel1);
    kernels.push_back(&basekernel2);
    kernels.push_back(&basekernel3);
    kernels.push_back(&basekernel4);
    kernels.push_back(&basekernel5);
    kernels.push_back(&basekernel6);
    DenseWeightedSumKernel kernel(kernels);

    // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    unsigned int num_bools = 5000;
    unsigned int num_trials = 50;
    std::vector< bool > cur_bools(numker);
    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        for ( unsigned int j=0; j<num_trials; j++ ) {
            RealVector cur_params(kernel.numberOfParameters());
            for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                cur_params(k) = Rng::discrete(1,10);
            kernel.setParameterVector(cur_params);
            BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        }
    }
    kernel.setAdaptiveAll(true);
    // test setting and getting the entire parameter vector
    for ( unsigned int j=0; j<num_trials; j++ ) {
        RealVector cur_params(kernel.numberOfParameters());
        for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            cur_params(k) = Rng::uni(1.0,3.0);
        kernel.setParameterVector(cur_params);
        BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.parameterVector()(0)-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.gamma()-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(8) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(9) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(10) , 1.e-15);
    }

    // test kernel evals on a test point
    RealVector my_params(11);
    my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1;
    my_params(7) = 0.5; my_params(8) = 0.6; my_params(9) = 2.0; my_params(10) = -1.1;
    kernel.setParameterVector(my_params);

    RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;

    double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    double k4 = k3*k3*k3;
    double k5 = 1.0;
    for ( unsigned int i=0; i<basekernel5.degree(); i++ ) k5 *= (k3+0.5);
    double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    double k = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    BOOST_CHECK_SMALL( k/d - kernel.eval(test1,test2), 1e-15 );

    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    }

    //test first derivative
    testKernelDerivative(kernel, 3, 1.e-5, 1.e-5);
    testKernelInputDerivative(kernel, 3, 1.e-5);
}

BOOST_AUTO_TEST_CASE( DenseWeightedSumKernel_Test_Detailed_NoDegreeParam_Unconstrained )
{
    unsigned int numdim = 3;
    unsigned int numker = 6;
    DenseRbfKernel        basekernel1(0.1);
    DenseRbfKernel        basekernel2(0.4, true);
    DenseLinearKernel     basekernel3;
    DenseMonomialKernel   basekernel4(3);
    DensePolynomialKernel basekernel5(2, 1.0, false, true);
    BOOST_CHECK( basekernel5.hasFirstParameterDerivative() );
    DenseARDKernel        basekernel6(numdim,0.2);

    std::vector< AbstractKernelFunction<RealVector> * > kernels;
    kernels.push_back(&basekernel1);
    kernels.push_back(&basekernel2);
    kernels.push_back(&basekernel3);
    kernels.push_back(&basekernel4);
    kernels.push_back(&basekernel5);
    kernels.push_back(&basekernel6);
    DenseWeightedSumKernel kernel(kernels);

    // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    unsigned int num_bools = 5000;
    unsigned int num_trials = 50;
    std::vector< bool > cur_bools(numker);
    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        for ( unsigned int j=0; j<num_trials; j++ ) {
            RealVector cur_params(kernel.numberOfParameters());
            for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                cur_params(k) = Rng::discrete(1,10);
            kernel.setParameterVector(cur_params);
            BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        }
    }
    kernel.setAdaptiveAll(true);
    // test setting and getting the entire parameter vector
    for ( unsigned int j=0; j<num_trials; j++ ) {
        RealVector cur_params(kernel.numberOfParameters());
        for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            cur_params(k) = Rng::uni(1.0,3.0);
        kernel.setParameterVector(cur_params);
        BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.parameterVector()(0)-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel1.gamma()-cur_params(5) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(8) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(9) , 1.e-15);
        BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(10) , 1.e-15);
    }

    // test kernel evals on a test point
    RealVector my_params(11);
    my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1;
    my_params(7) = -0.5; my_params(8) = 0.6; my_params(9) = 2.0; my_params(10) = -1.1;
    kernel.setParameterVector(my_params);

    RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;

    double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    double k4 = k3*k3*k3;
    double k5 = 1.0;
    for ( unsigned int i=0; i<basekernel5.degree(); i++ ) k5 *= (k3+std::exp(-0.5));
    double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
    BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    double k = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    BOOST_CHECK_SMALL( k/d - kernel.eval(test1,test2), 1e-15 );

    for ( unsigned int i=0; i<num_bools; i++ ) {
        for ( unsigned int k=0; k<numker; k++ ) {
            cur_bools[k] = Rng::discrete();
            kernel.setAdaptive(k, cur_bools[k]);
            BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        }
        BOOST_CHECK_SMALL( k1 - basekernel1.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k2 - basekernel2.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k3 - basekernel3.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k4 - basekernel4.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k5 - basekernel5.eval(test1,test2), 1e-15 );
        BOOST_CHECK_SMALL( k6 - basekernel6.eval(test1,test2), 1e-15 );
    }

    //test first derivative
    testKernelDerivative(kernel, 3, 1.e-5, 1.e-5);
    testKernelInputDerivative(kernel, 3, 1.e-5);
}

//BOOST_AUTO_TEST_CASE( DenseFullyWeightedSumKernel_Test )
//{
//  const double gamma1 = 0.1;
//  const double gamma2 = 0.01;
//  RealVector testParams(3);
//  testParams(0) = 0;
//  testParams(1) = 1;
//  testParams(2) = gamma2;

//  DenseRbfKernel baseKernel1(gamma1);
//  DenseRbfKernel baseKernel2(2*gamma2);
//  std::vector<AbstractKernelFunction<RealVector>* > kernels;
//  kernels.push_back(&baseKernel1);
//  kernels.push_back(&baseKernel2);
//  DenseFullyWeightedSumKernel kernel(kernels);
//  kernel.setAdaptive(1, true);

//  //now test whether the parametervector is created correctly
//  kernel.setParameterVector(testParams);
//  RealVector parameter=kernel.parameterVector();
//  BOOST_CHECK_SMALL(norm_sqr(parameter-testParams), 1.e-15);
//  //and check whether all gamma values are correct
//  BOOST_CHECK_SMALL(baseKernel1.gamma() - gamma1, 1e-13);
//  BOOST_CHECK_SMALL(baseKernel2.gamma() - gamma2, 1e-13);

//  //testpoints
//  RealVector x1(2);
//  x1(0)=2;
//  x1(1)=1;
//  RealVector x2(2);
//  x2(0)=-2;
//  x2(1)=1;

//  Intermediate intermediateK1;
//  Intermediate intermediateK2;
//  double k1 = baseKernel1.eval(x1,x2,intermediateK1);
//  double k2 = baseKernel2.eval(x1,x2,intermediateK2);
//  double numeratorResult = k1 + boost::math::constants::e<double>() * k2;
//  double result = numeratorResult / (1 + boost::math::constants::e<double>());
//
//  //evaluate point
//  double test = kernel.eval(x1,x2);
//  Intermediate intermediate;
//  double test2 = kernel.eval(x1,x2,intermediate);
//  BOOST_REQUIRE_SMALL(result - test, 1.e-15);
//  BOOST_REQUIRE_SMALL(result - test2, 1.e-15);
//
//  //test intermediate values everything is required to be correct because testing
//  //the derivative does not make sense anymore, if this is wrong
//  BOOST_REQUIRE_EQUAL(intermediate.size() , 7);
//  BOOST_REQUIRE_SMALL(intermediate[0] - numeratorResult,1.e-15);
//  BOOST_REQUIRE_SMALL(intermediate[1] - k1,1.e-15);
//  BOOST_REQUIRE_SMALL(intermediate[2] - intermediateK1[0],1.e-15);
//  BOOST_REQUIRE_SMALL(intermediate[3] - intermediateK1[1],1.e-15);
//  BOOST_REQUIRE_SMALL(intermediate[4] - k2,1.e-15);
//  BOOST_REQUIRE_SMALL(intermediate[5] - intermediateK2[0],1.e-15);
//  BOOST_REQUIRE_SMALL(intermediate[6] - intermediateK2[1],1.e-15);

//  //test first derivative
//  testKernelDerivative(kernel, 2, 1.e-6);
//  testKernelInputDerivative(kernel, 2, 1.e-8);
//}

BOOST_AUTO_TEST_SUITE_END()
