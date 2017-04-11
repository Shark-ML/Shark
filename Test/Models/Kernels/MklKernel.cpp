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

//adapt both struct to make boost fusion compatible, this is needed for eval() with single elements
BOOST_FUSION_ADAPT_STRUCT(
    TestStruct1,
    (shark::RealVector, v1)(std::size_t, v2)(shark::RealVector, v3)
)


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
            dataV1[i](j)=data[i].v1(j)=random::uni(random::globalRng,-1,1);
        }
        dataV2[i]=data[i].v2=random::discrete(random::globalRng,0,5);
        for(std::size_t j = 0; j != dim2; ++j){
            dataV3[i](j)=data[i].v3(j)=random::uni(random::globalRng,-1,1);
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
            matK(i,j) = random::uni(random::globalRng,-1,1);
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
    RealVector parameters = {0.3,0.1,0.3}; //weights are 1, 0.5 and 0.1 and the gauss kernel parameter is 0.3
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

BOOST_AUTO_TEST_SUITE_END()
