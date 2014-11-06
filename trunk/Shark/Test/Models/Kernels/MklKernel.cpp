#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/DiscreteKernel.h>
#include <shark/Models/Kernels/MklKernel.h>
#include <shark/Models/Kernels/KernelHelpers.h>

struct TestStruct1{
    shark::RealVector v1;
    std::size_t v2;//used as input for discrete kernel
    shark::RealVector v3;
};


//~ struct TestStruct2{//differentiable.
    //~ RealVector v1;
    //~ RealVector v2;
//~ };

//adapt both struct to make boost fusion compatible, this is needed for eval() with single elements
BOOST_FUSION_ADAPT_STRUCT(
    TestStruct1,
    (shark::RealVector, v1)(std::size_t, v2)(shark::RealVector, v3)
)
//~ BOOST_FUSION_ADAPT_STRUCT(
    //~ TestStruct2,
    //~ (RealVector, v1)(RealVector, v2)
//~ )

//Now adapt both structs to the batch interface
//todo make this less cumbersome. more like above.
namespace shark{
template<>
struct Batch< TestStruct1 >{
    SHARK_CREATE_BATCH_INTERFACE_NO_TPL(
        TestStruct1,
        (shark::RealVector, v1)(std::size_t, v2)(shark::RealVector, v3)
    )
};
//~ template<>
//~ struct Batch< TestStruct2 >{
    //~ SHARK_CREATE_BATCH_INTERFACE(
        //~ TestStruct2,
        //~ (RealVector, v1)(RealVector, v2)
    //~ )
//~ };
}
//not sure whther the definitions above can also go below...
#define BOOST_TEST_MODULE Kernels_MklKernel
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
using namespace shark;



// Since the MKLKernel is based on the weighted sum kernel (which is properly
// tested), we do not need to do numerical testing.
// However, we instantiate every function to check for compile errors and that
// eval() matches the hand-evaluated kernels.
BOOST_AUTO_TEST_SUITE (Models_Kernels_MklKernel)

BOOST_AUTO_TEST_CASE( DenseMklKernel_Test_Eval )
{

    //create Data
    std::size_t const examples = 1;
    std::size_t const dim1 = 10;
    std::size_t const dim2 = 12;
    std::size_t maxElem = 5;
    std::vector<TestStruct1> data(examples);
    std::vector<RealVector> dataV1(examples,RealVector(dim1));
    std::vector<std::size_t> dataV2(examples);
    std::vector<RealVector> dataV3(examples,RealVector(dim2));
    for(std::size_t i = 0; i != examples; ++i){
        data[i].v1.resize(dim1); data[i].v3.resize(dim2);

        for(std::size_t j = 0; j != dim1; ++j){
            dataV1[i](j)=data[i].v1(j)=Rng::uni(-1,1);
        }
        dataV2[i]=data[i].v2=Rng::discrete(0,5);
        for(std::size_t j = 0; j != dim2; ++j){
            dataV3[i](j)=data[i].v3(j)=Rng::uni(-1,1);
        }
    }
    //wouldn't it be nice if we could split these, too?
    Data<TestStruct1>   dataset = createDataFromRange( data, 10 );
    Data<RealVector>  datasetV1 = createDataFromRange( dataV1, 10 );
    Data<std::size_t> datasetV2 = createDataFromRange( dataV2, 10 );
    Data<RealVector>  datasetV3 = createDataFromRange( dataV3, 10 );

    //create MKL Kernel
    //create state matrix for the discrete kernel
    RealMatrix matK(maxElem+1,maxElem+1);
    for(std::size_t i = 0; i != matK.size1();++i){
        for(std::size_t j = 0; j <= i;++j){
            matK(i,j) = Rng::uni(-1,1);
            matK(j,i) = matK(i,j);
        }
    }
    DenseRbfKernel  baseKernelV1(0.1);
    DiscreteKernel baseKernelV2(matK);
    DenseLinearKernel  baseKernelV3;

    MklKernel<TestStruct1> kernel( boost::fusion::make_vector(&baseKernelV1,&baseKernelV2,&baseKernelV3) );

    //check correct number of parameters
    const unsigned int numParameters = 3;
    kernel.setAdaptiveAll(true);
    BOOST_REQUIRE_EQUAL(kernel.numberOfParameters(),numParameters);

    // test kernel evals. first set weighting factors
    RealVector parameters(3);
    init(parameters)<<0.3,0.1,0.3; //weights are 1, 0.5 and 0.1 and the gauss kernel parameter is 0.3
    kernel.setParameterVector(parameters);

    //process kernel matrices for each element separately and weight the results to get ground-truth data
    RealMatrix matV1 = calculateRegularizedKernelMatrix(baseKernelV1,datasetV1);
    RealMatrix matV2 = calculateRegularizedKernelMatrix(baseKernelV2,datasetV2);
    RealMatrix matV3 = calculateRegularizedKernelMatrix(baseKernelV3,datasetV3);
    RealMatrix kernelMatTest = matV1+std::exp(0.3)*matV2+std::exp(0.1)*matV3;
    kernelMatTest /=1+std::exp(0.3)+std::exp(0.1);

    //now calculate the kernel matrix of the MKL Kernel. it should be the same.
    RealMatrix kernelMat = calculateRegularizedKernelMatrix(kernel,dataset);

    //test
    for(std::size_t i = 0; i != examples; ++i){
        for(std::size_t j = 0; j != examples;++j){
            BOOST_CHECK_CLOSE(kernelMatTest(i,j),kernelMat(i,j),1.e-5);
        }
    }
}

//~ // Test the MKL kernel with all sub-ranges exactly the full range
//~ BOOST_AUTO_TEST_CASE( DenseMklKernel_Test_FullRange )
//~ {
    //~ const double gamma1 = 0.1;
    //~ const double gamma2 = 0.01;
    //~ RealVector testParams(2);
    //~ testParams(0) = 1;
    //~ testParams(1) = gamma2;

    //~ GaussianRbfKernel<ConstRealVectorRange> baseKernelV1(gamma1);
    //~ GaussianRbfKernel<ConstRealVectorRange> baseKernel2(2*gamma2);
    //~ std::vector<AbstractKernelFunction<ConstRealVectorRange>* > kernels;
    //~ kernels.push_back(&baseKernelV1);
    //~ kernels.push_back(&baseKernel2);
    //~ std::vector< std::pair< std::size_t, std::size_t > > frs;
    //~ frs.push_back( std::make_pair( 0,2 ) );
    //~ frs.push_back( std::make_pair( 0,2 ) );
    //~ DenseMklKernel kernel( kernels, frs );
    //~ kernel.setAdaptive(1, true);

    //~ //now test whether the parametervector is created correctly
    //~ kernel.setParameterVector(testParams);
    //~ RealVector parameter=kernel.parameterVector();
    //~ BOOST_CHECK_SMALL(norm_sqr(parameter-testParams), 1.e-15);
    //~ //and check whether all gamma values are correct
    //~ BOOST_CHECK_SMALL(baseKernelV1.gamma() - gamma1, 1e-13);
    //~ BOOST_CHECK_SMALL(baseKernel2.gamma() - gamma2, 1e-13);

    //~ //testpoints
    //~ RealVector x1(2); x1(0)= 2; x1(1)=1;
    //~ RealVector x2(2); x2(0)=-2; x2(1)=1;
    //~ ConstRealVectorRange sub1 = subrange((const RealVector&)(x1),0,2);
    //~ ConstRealVectorRange sub2 = subrange((const RealVector&)(x2),0,2);

    //~ Intermediate intermediateK1;
    //~ Intermediate intermediateK2;
    //~ double k1 = baseKernelV1.eval( sub1, sub2, intermediateK1 );
    //~ double k2 = baseKernel2.eval( sub1, sub2, intermediateK2 );
    //~ double numeratorResult = k1 + boost::math::constants::e<double>() * k2;
    //~ double result = numeratorResult / ( 1+boost::math::constants::e<double>() );

    //~ //evaluate point
    //~ double test = kernel.eval(x1,x2);
    //~ Intermediate intermediate;
    //~ double test2 = kernel.eval(x1,x2,intermediate);
    //~ BOOST_REQUIRE_SMALL( result-test, 1.e-15 );
    //~ BOOST_REQUIRE_SMALL( result-test2, 1.e-15 );

    //~ //test intermediate values. everything is required to be correct because testing
    //~ //the derivative does not make sense anymore, if this is wrong
    //~ BOOST_REQUIRE_EQUAL(intermediate.size() , 7);
    //~ BOOST_REQUIRE_SMALL(intermediate[0] - numeratorResult,1.e-15);
    //~ BOOST_REQUIRE_SMALL(intermediate[1] - k1,1.e-15);
    //~ BOOST_REQUIRE_SMALL(intermediate[2] - intermediateK1[0],1.e-15);
    //~ BOOST_REQUIRE_SMALL(intermediate[3] - intermediateK1[1],1.e-15);
    //~ BOOST_REQUIRE_SMALL(intermediate[4] - k2,1.e-15);
    //~ BOOST_REQUIRE_SMALL(intermediate[5] - intermediateK2[0],1.e-15);
    //~ BOOST_REQUIRE_SMALL(intermediate[6] - intermediateK2[1],1.e-15);

    //~ //test first derivative
    //~ testKernelDerivative(kernel, 2, 1.e-6);
    //~ testKernelInputDerivative(kernel, 2, 1.e-6);
//~ }

//~ BOOST_AUTO_TEST_CASE( DenseMklKernel_Test_Detailed )
//~ {
    //~ unsigned int numdim = 3;
    //~ unsigned int numker = 6;
    //~ DenseRbfMklKernel        baseKernelV1(0.1);
    //~ DenseRbfMklKernel        basekernel2(0.4, true);
    //~ DenseLinearMklKernel     basekernel3;
    //~ DenseMonomialMklKernel   basekernel4(3);
    //~ DensePolynomialMklKernel basekernel5(2, 1.0);
    //~ BOOST_CHECK( !basekernel5.hasFirstParameterDerivative() );
    //~ DenseARDMklKernel         basekernel6(numdim,0.2);

    //~ std::vector< DenseMklKernelFunction * > kernels;
    //~ kernels.push_back(&baseKernelV1);
    //~ kernels.push_back(&basekernel2);
    //~ kernels.push_back(&basekernel3);
    //~ kernels.push_back(&basekernel4);
    //~ kernels.push_back(&basekernel5);
    //~ kernels.push_back(&basekernel6);
    //~ std::vector< std::pair< std::size_t, std::size_t > > frs;
    //~ for ( std::size_t i=0; i<6; i++ )
        //~ frs.push_back( std::make_pair( 0,3 ) );
    //~ DenseMklKernel kernel( kernels, frs );
    //~ BOOST_CHECK_SMALL( (double)kernel.numberOfIntermediateValues(RealVector(), RealVector()) - 16.0, 1e-15 );

    //~ // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    //~ unsigned int num_bools = 5000;
    //~ unsigned int num_trials = 50;
    //~ std::vector< bool > cur_bools(numker);
    //~ for ( unsigned int i=0; i<num_bools; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ for ( unsigned int j=0; j<num_trials; j++ ) {
            //~ RealVector cur_params(kernel.numberOfParameters());
            //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                //~ cur_params(k) = Rng::discrete(1,10);
            //~ kernel.setParameterVector(cur_params);
            //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        //~ }
    //~ }
    //~ kernel.setAdaptiveAll(true);
    //~ // test setting and getting the entire parameter vector
    //~ for ( unsigned int j=0; j<num_trials; j++ ) {
        //~ RealVector cur_params(kernel.numberOfParameters());
        //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            //~ cur_params(k) = Rng::uni(1.0,3.0);
        //~ cur_params(7) = (unsigned int)cur_params(7);
        //~ kernel.setParameterVector(cur_params);
        //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        //~ BOOST_CHECK_SMALL( baseKernelV1.parameterVector()(0)-cur_params(5) , 1.e-15);
        //~ BOOST_CHECK_SMALL( baseKernelV1.gamma()-cur_params(5) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel5.parameterVector()(1)-cur_params(8) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(9) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(10) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(11) , 1.e-15);
    //~ }

    //~ // test kernel evals on a test point
    //~ RealVector my_params(12);
    //~ my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    //~ my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1; my_params(7) = 3;
    //~ my_params(8) = 0.5; my_params(9) = 0.6; my_params(10) = 2.0; my_params(11) = -1.1;
    //~ kernel.setParameterVector(my_params);

    //~ RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    //~ RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;
    //~ ConstRealVectorRange sub1 = subrange((const RealVector&)(test1),0,3);
    //~ ConstRealVectorRange sub2 = subrange((const RealVector&)(test2),0,3);

    //~ double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    //~ double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    //~ double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    //~ double k4 = k3*k3*k3;
    //~ double k5 = (k3+0.5)*(k3+0.5)*(k3+0.5);
    //~ double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
    //~ double k = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    //~ double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    //~ BOOST_CHECK_SMALL( k/d - kernel.eval(test1,test2), 1e-15 );

    //~ for ( unsigned int i=0; i<num_bools; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
    //~ }
//~ }

//~ BOOST_AUTO_TEST_CASE( DenseMklKernel_Test_Detailed_Unconstrained )
//~ {
    //~ unsigned int numdim = 3;
    //~ unsigned int numker = 6;
    //~ DenseRbfMklKernel        baseKernelV1(0.1);
    //~ DenseRbfMklKernel        basekernel2(0.4, true);
    //~ DenseLinearMklKernel     basekernel3;
    //~ DenseMonomialMklKernel   basekernel4(3);
    //~ DensePolynomialMklKernel basekernel5(2, 1.0, true, true);
    //~ BOOST_CHECK( !basekernel5.hasFirstParameterDerivative() );
    //~ DenseARDMklKernel         basekernel6(numdim,0.2);

    //~ std::vector< DenseMklKernelFunction * > kernels;
    //~ kernels.push_back(&baseKernelV1);
    //~ kernels.push_back(&basekernel2);
    //~ kernels.push_back(&basekernel3);
    //~ kernels.push_back(&basekernel4);
    //~ kernels.push_back(&basekernel5);
    //~ kernels.push_back(&basekernel6);
    //~ std::vector< std::pair< std::size_t, std::size_t > > frs;
    //~ for ( std::size_t i=0; i<6; i++ )
        //~ frs.push_back( std::make_pair( 0,3 ) );
    //~ DenseMklKernel kernel( kernels, frs );
    //~ BOOST_CHECK_SMALL( (double)kernel.numberOfIntermediateValues(RealVector(), RealVector()) - 16.0, 1e-15 );

    //~ // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    //~ unsigned int num_bools = 5000;
    //~ unsigned int num_trials = 50;
    //~ std::vector< bool > cur_bools(numker);
    //~ for ( unsigned int i=0; i<num_bools; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ for ( unsigned int j=0; j<num_trials; j++ ) {
            //~ RealVector cur_params(kernel.numberOfParameters());
            //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                //~ cur_params(k) = Rng::discrete(1,10);
            //~ kernel.setParameterVector(cur_params);
            //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        //~ }
    //~ }
    //~ kernel.setAdaptiveAll(true);
    //~ // test setting and getting the entire parameter vector
    //~ for ( unsigned int j=0; j<num_trials; j++ ) {
        //~ RealVector cur_params(kernel.numberOfParameters());
        //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            //~ cur_params(k) = Rng::uni(1.0,3.0);
        //~ cur_params(7) = (unsigned int)cur_params(7);
        //~ kernel.setParameterVector(cur_params);
        //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        //~ BOOST_CHECK_SMALL( baseKernelV1.parameterVector()(0)-cur_params(5) , 1.e-15);
        //~ BOOST_CHECK_SMALL( baseKernelV1.gamma()-cur_params(5) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel5.parameterVector()(1)-cur_params(8) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(9) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(10) , 1.e-15);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(11) , 1.e-15);
    //~ }

    //~ // test kernel evals on a test point
    //~ RealVector my_params(12);
    //~ my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    //~ my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1; my_params(7) = 3;
    //~ my_params(8) = -0.5; my_params(9) = 0.6; my_params(10) = 2.0; my_params(11) = -1.1;
    //~ kernel.setParameterVector(my_params);

    //~ RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    //~ RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;
    //~ ConstRealVectorRange sub1 = subrange((const RealVector&)(test1),0,3);
    //~ ConstRealVectorRange sub2 = subrange((const RealVector&)(test2),0,3);

    //~ double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    //~ double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    //~ double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    //~ double k4 = k3*k3*k3;
    //~ double k5 = (k3+std::exp(-0.5))*(k3+std::exp(-0.5))*(k3+std::exp(-0.5));
    //~ double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
    //~ double k = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    //~ double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    //~ BOOST_CHECK_SMALL( k/d - kernel.eval(test1,test2), 1e-15 );

    //~ for ( unsigned int i=0; i<num_bools; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
    //~ }
//~ }

//~ BOOST_AUTO_TEST_CASE( DenseMklKernel_Test_Detailed_NoDegreeParam )
//~ {
    //~ unsigned int numdim = 3;
    //~ unsigned int numker = 6;
    //~ DenseRbfMklKernel         baseKernelV1(0.1);
    //~ DenseRbfMklKernel         basekernel2(0.4, true);
    //~ DenseLinearMklKernel     basekernel3;
    //~ DenseMonomialMklKernel   basekernel4(3);
    //~ DensePolynomialMklKernel basekernel5(2, 1.0, false);
    //~ BOOST_CHECK( basekernel5.hasFirstParameterDerivative() );
    //~ basekernel5.setDegree( 3 );
//~ //  basekernel5.setDegree( 0 ); //should fail
    //~ BOOST_CHECK_EQUAL( basekernel5.degree(), 3 );
    //~ basekernel5.setDegree( 2 );
    //~ DenseARDMklKernel         basekernel6(numdim,0.2);

    //~ std::vector< DenseMklKernelFunction * > kernels;
    //~ kernels.push_back(&baseKernelV1);
    //~ kernels.push_back(&basekernel2);
    //~ kernels.push_back(&basekernel3);
    //~ kernels.push_back(&basekernel4);
    //~ kernels.push_back(&basekernel5);
    //~ kernels.push_back(&basekernel6);
    //~ std::vector< std::pair< std::size_t, std::size_t > > frs;
    //~ for ( std::size_t i=0; i<6; i++ )
        //~ frs.push_back( std::make_pair( 0,3 ) );
    //~ DenseMklKernel kernel( kernels, frs );
    //~ BOOST_CHECK_SMALL( (double)kernel.numberOfIntermediateValues(RealVector(), RealVector()) - 16.0, 1e-15 );

    //~ // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    //~ unsigned int num_bools = 5000;
    //~ unsigned int num_trials = 50;
    //~ std::vector< bool > cur_bools(numker);
    //~ for ( unsigned int i=0; i<num_bools; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ for ( unsigned int j=0; j<num_trials; j++ ) {
            //~ RealVector cur_params(kernel.numberOfParameters());
            //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                //~ cur_params(k) = Rng::discrete(1,10);
            //~ kernel.setParameterVector(cur_params);
            //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        //~ }
    //~ }
    //~ kernel.setAdaptiveAll(true);
    //~ // test setting and getting the entire parameter vector
    //~ for ( unsigned int j=0; j<num_trials; j++ ) {
        //~ RealVector cur_params(kernel.numberOfParameters());
        //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            //~ cur_params(k) = Rng::uni(1.0,3.0);
        //~ kernel.setParameterVector(cur_params);
        //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-14);
        //~ BOOST_CHECK_SMALL( baseKernelV1.parameterVector()(0)-cur_params(5) , 1.e-14);
        //~ BOOST_CHECK_SMALL( baseKernelV1.gamma()-cur_params(5) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(8) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(9) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(10) , 1.e-14);
    //~ }

    //~ // test kernel evals on a test point
    //~ RealVector my_params(11);
    //~ my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    //~ my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1;
    //~ my_params(7) = 0.5; my_params(8) = 0.6; my_params(9) = 2.0; my_params(10) = -1.1;
    //~ kernel.setParameterVector(my_params);

    //~ RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    //~ RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;
    //~ ConstRealVectorRange sub1 = subrange((const RealVector&)(test1),0,3);
    //~ ConstRealVectorRange sub2 = subrange((const RealVector&)(test2),0,3);

    //~ double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    //~ double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    //~ double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    //~ double k4 = k3*k3*k3;
    //~ double k5 = 1.0;
    //~ for ( unsigned int i=0; i<basekernel5.degree(); i++ ) k5 *= (k3+0.5);
    //~ double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
    //~ double kk = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    //~ double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    //~ BOOST_CHECK_SMALL( kk/d - kernel.eval(test1,test2), 1e-15 );

    //~ for ( unsigned int i=0; i<250; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( kk/d - kernel.eval(test1,test2), 1e-15 );
        //~ testKernelDerivative(kernel, 3, 1.e-5, 1.e-5, 50);
        //~ testKernelInputDerivative(kernel, 3, 1.e-5, 1.e-5, 50);
    //~ }
    //~ kernel.setAdaptiveAll(true);

    //~ //test first derivative
    //~ kernel.setAdaptiveAll(true);
    //~ testKernelDerivative(kernel, 3, 1.e-5, 1.e-5);
    //~ testKernelInputDerivative(kernel, 3, 1.e-5);
//~ }

//~ BOOST_AUTO_TEST_CASE( DenseMklKernel_Test_Detailed_NoDegreeParam_Unconstrained )
//~ {
    //~ unsigned int numdim = 3;
    //~ unsigned int numker = 6;
    //~ DenseRbfMklKernel         baseKernelV1(0.1);
    //~ DenseRbfMklKernel         basekernel2(0.4, true);
    //~ DenseLinearMklKernel      basekernel3;
    //~ DenseMonomialMklKernel    basekernel4(3);
    //~ DensePolynomialMklKernel  basekernel5(2, 1.0, false, true);
    //~ BOOST_CHECK( basekernel5.hasFirstParameterDerivative() );
    //~ DenseARDMklKernel         basekernel6(numdim,0.2);

    //~ std::vector< DenseMklKernelFunction * > kernels;
    //~ kernels.push_back(&baseKernelV1);
    //~ kernels.push_back(&basekernel2);
    //~ kernels.push_back(&basekernel3);
    //~ kernels.push_back(&basekernel4);
    //~ kernels.push_back(&basekernel5);
    //~ kernels.push_back(&basekernel6);
    //~ std::vector< std::pair< std::size_t, std::size_t > > frs;
    //~ for ( std::size_t i=0; i<6; i++ )
        //~ frs.push_back( std::make_pair( 0,3 ) );
    //~ DenseMklKernel kernel( kernels, frs );
    //~ BOOST_CHECK_SMALL( (double)kernel.numberOfIntermediateValues(RealVector(), RealVector()) - 16.0, 1e-15 );

    //~ // test setting and getting the parameter vector for all kinds of adaptive-ness scenarios
    //~ unsigned int num_bools = 5000;
    //~ unsigned int num_trials = 50;
    //~ std::vector< bool > cur_bools(numker);
    //~ for ( unsigned int i=0; i<num_bools; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ for ( unsigned int j=0; j<num_trials; j++ ) {
            //~ RealVector cur_params(kernel.numberOfParameters());
            //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
                //~ cur_params(k) = Rng::discrete(1,10);
            //~ kernel.setParameterVector(cur_params);
            //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-15);
        //~ }
    //~ }
    //~ kernel.setAdaptiveAll(true);
    //~ // test setting and getting the entire parameter vector
    //~ for ( unsigned int j=0; j<num_trials; j++ ) {
        //~ RealVector cur_params(kernel.numberOfParameters());
        //~ for ( unsigned int k=0; k<kernel.numberOfParameters(); k++ )
            //~ cur_params(k) = Rng::uni(1.0,3.0);
        //~ kernel.setParameterVector(cur_params);
        //~ BOOST_CHECK_SMALL(norm_sqr(kernel.parameterVector()-cur_params), 1.e-14);
        //~ BOOST_CHECK_SMALL( baseKernelV1.parameterVector()(0)-cur_params(5) , 1.e-14);
        //~ BOOST_CHECK_SMALL( baseKernelV1.gamma()-cur_params(5) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel2.parameterVector()(0)-cur_params(6) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel2.gamma()-std::exp(cur_params(6)) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel5.parameterVector()(0)-cur_params(7) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(0)-cur_params(8) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(1)-cur_params(9) , 1.e-14);
        //~ BOOST_CHECK_SMALL( basekernel6.parameterVector()(2)-cur_params(10) , 1.e-14);
    //~ }

    //~ // test kernel evals on a test point
    //~ RealVector my_params(11);
    //~ my_params(0) = 0.2; my_params(1) = 0.4; my_params(2) = 0.0; my_params(3) = 1.0;
    //~ my_params(4) = -1.0; my_params(5) = 0.1; my_params(6) = -0.1;
    //~ my_params(7) = -0.5; my_params(8) = 0.6; my_params(9) = 2.0; my_params(10) = -1.1;
    //~ kernel.setParameterVector(my_params);

    //~ RealVector test1(3); test1(0) = 1.1; test1(1) = 0.7; test1(2) = -1.3;
    //~ RealVector test2(3); test2(0) = -0.7; test2(1) = 2.1; test2(2) = 0.1;
    //~ ConstRealVectorRange sub1 = subrange((const RealVector&)(test1),0,3);
    //~ ConstRealVectorRange sub2 = subrange((const RealVector&)(test2),0,3);

    //~ double k1 = std::exp( -0.1*(1.8*1.8 + 1.4*1.4 + 1.4*1.4)  );
    //~ double k2 = std::exp( -std::exp(-0.1)*( (1.8*1.8 + 1.4*1.4 + 1.4*1.4) ) );
    //~ double k3 = -1.1*0.7 + 0.7*2.1 + -1.3*0.1;
    //~ double k4 = k3*k3*k3;
    //~ double k5 = 1.0;
    //~ for ( unsigned int i=0; i<basekernel5.degree(); i++ ) k5 *= (k3+std::exp(-0.5));
    //~ double k6 = std::exp( -0.6*0.6*1.8*1.8 - 2.0*2.0*1.4*1.4 -1.1*1.1*1.4*1.4 );
    //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
    //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
    //~ double kk = std::exp(0.0)*k1 + std::exp(0.2)*k2 + std::exp(0.4)*k3 + std::exp(0.0)*k4 + std::exp(1.0)*k5 + std::exp(-1.0)*k6;
    //~ double d = std::exp(0.0) + std::exp(0.2) + std::exp(0.4) + std::exp(0.0) + std::exp(1.0) + std::exp(-1.0);
    //~ BOOST_CHECK_SMALL( kk/d - kernel.eval(test1,test2), 1e-15 );

    //~ for ( unsigned int i=0; i<250; i++ ) {
        //~ for ( unsigned int k=0; k<numker; k++ ) {
            //~ cur_bools[k] = Rng::discrete();
            //~ kernel.setAdaptive(k, cur_bools[k]);
            //~ BOOST_REQUIRE_EQUAL( cur_bools[k], kernel.isAdaptive(k) );
        //~ }
        //~ BOOST_CHECK_SMALL( k1 - baseKernelV1.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k2 - basekernel2.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k3 - basekernel3.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k4 - basekernel4.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k5 - basekernel5.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( k6 - basekernel6.eval(sub1,sub2), 1e-15 );
        //~ BOOST_CHECK_SMALL( kk/d - kernel.eval(test1,test2), 1e-15 );
        //~ testKernelDerivative(kernel, 3, 1.e-5, 1.e-5, 50);
        //~ testKernelInputDerivative(kernel, 3, 1.e-5, 1.e-5, 50);
    //~ }

    //~ //test first derivative
    //~ kernel.setAdaptiveAll(true);
    //~ testKernelDerivative(kernel, 3, 1.e-5, 1.e-5);
    //~ testKernelInputDerivative(kernel, 3, 1.e-5);
//~ }

BOOST_AUTO_TEST_SUITE_END()
