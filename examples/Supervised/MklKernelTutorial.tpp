//###begin<includes_namespaces>
#include <shark/Data/Dataset.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/Algorithms/Trainers/NormalizeKernelUnitVariance.h>
#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/Models/Kernels/WeightedSumKernel.h>
#include <shark/Models/Kernels/SubrangeKernel.h>
#include <shark/Models/Kernels/MklKernel.h>
#include <shark/Models/Kernels/LinearKernel.h>
#include <shark/Models/Kernels/DiscreteKernel.h>
#include <shark/Models/Kernels/PolynomialKernel.h>
#include <boost/fusion/algorithm/iteration/fold.hpp>
#include <boost/fusion/include/as_vector.hpp>
//###end<includes_namespaces>

    //###begin<create_struct_batch_interface>
    struct HeterogeneousInputStruct{
        shark::RealVector rv1;
        std::size_t st2;
        shark::CompressedRealVector crv3;
    };

    #ifndef DOXYGEN_SHOULD_SKIP_THIS
        BOOST_FUSION_ADAPT_STRUCT(
            HeterogeneousInputStruct,
            (shark::RealVector, rv1)(std::size_t, st2)(shark::CompressedRealVector, crv3)
        )
    #endif /* DOXYGEN_SHOULD_SKIP_THIS */

    namespace shark{
        template<>
        struct Batch< HeterogeneousInputStruct >{
            SHARK_CREATE_BATCH_INTERFACE_NO_TPL(
                HeterogeneousInputStruct,
                (shark::RealVector, rv1)(std::size_t, st2)(shark::CompressedRealVector, crv3)
            )
        };
    }
    //###end<create_struct_batch_interface>

//###begin<includes_namespaces>
using namespace shark;
using namespace std;
//###end<includes_namespaces>




int main(int argc, char** argv)
{

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    //###begin<test_point_setup>
    // test points
    RealVector x1(2);
    x1(0)=2;
    x1(1)=1;
    RealVector x2(2);
    x2(0)=-2;
    x2(1)=1;
    //###end<test_point_setup>


    //###begin<kernel_setup>
    // initialize kernels
    DenseRbfKernel baseKernel1( 0.1 );
    DenseRbfKernel baseKernel2( 0.01 );
    std::vector< AbstractKernelFunction<RealVector> * > kernels1;
    kernels1.push_back( &baseKernel1 );
    kernels1.push_back( &baseKernel2 );
    DenseWeightedSumKernel kernel1( kernels1 );
    //###end<kernel_setup>

////////////////////////////////////////////////////////////////////////////////

    //###begin<introspection_one>
    // examine initial state
    std::cout << endl << " ======================= WeightedSumKernel: ======================= " << std::endl;
    cout << endl << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl;
    cout << "kernel1.eval(x1,x2): " << kernel1.eval(x1,x2) << endl << endl;
    //###end<introspection_one>

    //###begin<change_something_one>
    // change something
    RealVector new_params_1( kernel1.numberOfParameters() );
    new_params_1(0) = 1.0;
    kernel1.setParameterVector( new_params_1 );
    //###end<change_something_one>

    //###begin<introspection_two>
    // examine again
    cout << "kernel1.parameterVector() with 1st parameter set to 1: " << kernel1.parameterVector() << endl;
    cout << "kernel1.eval(x1,x2): " << kernel1.eval(x1,x2) << endl << endl;
    //###end<introspection_two>

    //###begin<change_something_two>
    // change something else
    kernel1.setAdaptive(0,true);
    //###end<change_something_two>

    //###begin<introspection_three>
    // examine once more
    cout << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl<< endl;
    //###end<introspection_three>

    //###begin<change_something_three>
    // another change
    kernel1.setAdaptive(0,false);
    kernel1.setAdaptive(1,true);
    //###end<change_something_three>

    //###begin<introspection_four>
    // examining again
    cout << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl<< endl;
    //###end<introspection_four>

    //###begin<change_something_four>
    // last change
    kernel1.setAdaptiveAll(true);
    //###end<change_something_four>

    //###begin<introspection_five>
    // last examination
    cout << "kernel1.isAdaptive(0): " << kernel1.isAdaptive(0) << endl;
    cout << "kernel1.isAdaptive(1): " << kernel1.isAdaptive(1) << endl;
    cout << "kernel1.numberOfParameters(): " << kernel1.numberOfParameters() << endl;
    cout << "kernel1.parameterVector(): " << kernel1.parameterVector() << endl;
    cout << "kernel1.eval(x1,x2): " << kernel1.eval(x1,x2) << endl << endl;
    //###end<introspection_five>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    //###begin<subrange_kernel_setup_1>
    DenseRbfKernel baseKernel3(0.1);
    DenseRbfKernel baseKernel4(0.01);
    std::vector<AbstractKernelFunction<RealVector>* > kernels2;
    kernels2.push_back(&baseKernel3);
    kernels2.push_back(&baseKernel4);
    //###end<subrange_kernel_setup_1>

    //###begin<subrange_indices_setup_1>
    std::vector< std::pair< std::size_t, std::size_t > > indcs_1;
    indcs_1.push_back( std::make_pair( 0,2 ) );
    indcs_1.push_back( std::make_pair( 0,2 ) );
    DenseSubrangeKernel kernel2( kernels2, indcs_1 );
    //###end<subrange_indices_setup_1>

////////////////////////////////////////////////////////////////////////////////

    //###begin<subrange_introspection_one>
    // examine initial state
    std::cout << endl << " ======================= SubrangeKernel, full index range: ======================= " << std::endl;
    cout << endl << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl;
    cout << "kernel2.eval(x1,x2): " << kernel2.eval(x1,x2) << endl << endl;
    //###end<subrange_introspection_one>

    //###begin<subrange_change_something_one>
    // change something
    RealVector new_params_2( kernel2.numberOfParameters() );
    new_params_2(0) = 1.0;
    kernel2.setParameterVector( new_params_2 );
    //###end<subrange_change_something_one>

    //###begin<subrange_introspection_two>
    // examine again
    cout << "kernel2.parameterVector() with 1st parameter set to 1: " << kernel2.parameterVector() << endl;
    cout << "kernel2.eval(x1,x2): " << kernel2.eval(x1,x2) << endl << endl;
    //###end<subrange_introspection_two>

    //###begin<subrange_change_something_two>
    // change something else
    kernel2.setAdaptive(0,true);
    //###end<subrange_change_something_two>

    //###begin<subrange_introspection_three>
    // examine once more
    cout << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl<< endl;
    //###end<subrange_introspection_three>

    //###begin<subrange_change_something_three>
    // another change
    kernel2.setAdaptive(0,false);
    kernel2.setAdaptive(1,true);
    //###end<subrange_change_something_three>

    //###begin<subrange_introspection_four>
    // examining again
    cout << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl<< endl;
    //###end<subrange_introspection_four>

    //###begin<subrange_change_something_four>
    // last change
    kernel2.setAdaptiveAll(true);
    //###end<subrange_change_something_four>

    //###begin<subrange_introspection_five>
    // last examination
    cout << "kernel2.isAdaptive(0): " << kernel2.isAdaptive(0) << endl;
    cout << "kernel2.isAdaptive(1): " << kernel2.isAdaptive(1) << endl;
    cout << "kernel2.numberOfParameters(): " << kernel2.numberOfParameters() << endl;
    cout << "kernel2.parameterVector(): " << kernel2.parameterVector() << endl;
    cout << "kernel2.eval(x1,x2): " << kernel2.eval(x1,x2) << endl << endl;
    //###end<subrange_introspection_five>


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


    //###begin<subrange_kernel_setup_2>
    DenseRbfKernel baseKernel5(0.1);
    DenseRbfKernel baseKernel6(0.01);
    std::vector<AbstractKernelFunction<RealVector>* > kernels3;
    kernels3.push_back(&baseKernel5);
    kernels3.push_back(&baseKernel6);
    //###end<subrange_kernel_setup_2>

    //###begin<subrange_indices_setup_2>
    std::vector< std::pair< std::size_t, std::size_t > > indcs_2;
    indcs_2.push_back( std::make_pair( 0,1 ) );
    indcs_2.push_back( std::make_pair( 1,2 ) );
    DenseSubrangeKernel kernel3( kernels3, indcs_2 );
    //###end<subrange_indices_setup_2>

////////////////////////////////////////////////////////////////////////////////

    //###begin<2_subrange_introspection_one>
    // examine initial state
    std::cout << endl << " ======================= SubrangeKernel partial index range: ======================= " << std::endl;
    cout << endl << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl;
    cout << "kernel3.eval(x1,x2): " << kernel3.eval(x1,x2) << endl << endl;
    //###end<2_subrange_introspection_one>

    //###begin<2_subrange_change_something_one>
    // change something
    RealVector new_params_3( kernel3.numberOfParameters() );
    new_params_3(0) = 1.0;
    kernel3.setParameterVector( new_params_3 );
    //###end<2_subrange_change_something_one>

    //###begin<2_subrange_introspection_two>
    // examine again
    cout << "kernel3.parameterVector() with 1st parameter set to 1: " << kernel3.parameterVector() << endl;
    cout << "kernel3.eval(x1,x2): " << kernel3.eval(x1,x2) << endl << endl;
    //###end<2_subrange_introspection_two>

    //###begin<2_subrange_change_something_two>
    // change something else
    kernel3.setAdaptive(0,true);
    //###end<2_subrange_change_something_two>

    //###begin<2_subrange_introspection_three>
    // examine once more
    cout << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl<< endl;
    //###end<2_subrange_introspection_three>

    //###begin<2_subrange_change_something_three>
    // another change
    kernel3.setAdaptive(0,false);
    kernel3.setAdaptive(1,true);
    //###end<2_subrange_change_something_three>

    //###begin<2_subrange_introspection_four>
    // examining again
    cout << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl<< endl;
    //###end<2_subrange_introspection_four>

    //###begin<2_subrange_change_something_four>
    // last change
    kernel3.setAdaptiveAll(true);
    //###end<2_subrange_change_something_four>

    //###begin<2_subrange_introspection_five>
    // last examination
    cout << "kernel3.isAdaptive(0): " << kernel3.isAdaptive(0) << endl;
    cout << "kernel3.isAdaptive(1): " << kernel3.isAdaptive(1) << endl;
    cout << "kernel3.numberOfParameters(): " << kernel3.numberOfParameters() << endl;
    cout << "kernel3.parameterVector(): " << kernel3.parameterVector() << endl;
    cout << "kernel3.eval(x1,x2): " << kernel3.eval(x1,x2) << endl << endl;
    //###end<2_subrange_introspection_five>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


    //###begin<mkl_kernel_fill_struct>
    // set dimensions for data
    std::size_t const num_samples = 2;
    std::size_t const dim_nonzeros = 2;
    std::size_t const max_elem_discr_kernel = 3;
    std::size_t const dim_sparse = 5;
    // create temporary helper container
    std::vector<HeterogeneousInputStruct> data( num_samples );
    // and fill it
    data[0].rv1.resize( dim_nonzeros ); data[0].crv3.resize( dim_sparse, dim_nonzeros ); //size 5 w/ 2 non-zeros
    data[1].rv1.resize( dim_nonzeros ); data[1].crv3.resize( dim_sparse, dim_nonzeros ); //size 5 w/ 2 non-zeros
    data[0].rv1(0) = 1.0; data[0].rv1(1) = -1.0; data[0].crv3(1) = -0.5; data[0].crv3(4) = 8.0;
    data[1].rv1(0) = 1.0; data[1].rv1(1) = -2.0; data[1].crv3(1) =  1.0; data[1].crv3(3) = 0.1;
    data[0].st2 = 1; data[1].st2 = 2;
    // and use it to create the 'real' dataset
    Data<HeterogeneousInputStruct> dataset = createDataFromRange( data, 10 );
    //###end<mkl_kernel_fill_struct>

////////////////////////////////////////////////////////////////////////////////

    //###begin<mkl_kernel_create_kernels>
    //create state matrix for the discrete kernel. necessary but not so relevant
    RealMatrix matK( max_elem_discr_kernel, max_elem_discr_kernel );
    matK(0,0) = 0.05; matK(1,1) = 1.0;  matK(2,2) = 0.5;
    matK(0,1) = matK(1,0) = 0.2; matK(0,2) = matK(2,0) = 0.4;  matK(1,2) = matK(2,1) = 0.6;
    // set up base kernels
    DenseRbfKernel baseKernelRV1(0.1);
    DiscreteKernel baseKernelST2(matK);
    CompressedLinearKernel baseKernelCRV3;
    MklKernel<HeterogeneousInputStruct> mkl_kernel( boost::fusion::make_vector( &baseKernelRV1, &baseKernelST2, &baseKernelCRV3) );
    //###end<mkl_kernel_create_kernels>

    //###begin<mkl_introspection_one>
    // examine initial state
    std::cout << endl << " ======================= MklKernel: ======================= " << std::endl;
    cout << endl << "mkl_kernel.isAdaptive(0): " << mkl_kernel.isAdaptive(0) << endl;
    cout << "mkl_kernel.isAdaptive(1): " << mkl_kernel.isAdaptive(1) << endl;
    cout << "mkl_kernel.isAdaptive(2): " << mkl_kernel.isAdaptive(2) << endl;
    cout << "mkl_kernel.numberOfParameters(): " << mkl_kernel.numberOfParameters() << endl;
    cout << "mkl_kernel.parameterVector(): " << mkl_kernel.parameterVector() << endl;
    cout << "mkl_kernel.eval( dataset.element(0), dataset.element(1) ): " << mkl_kernel.eval( dataset.element(0), dataset.element(1) ) << endl << endl;
    //###end<mkl_introspection_one>

    //###begin<mkl_change_something_one>
    // change something
    mkl_kernel.setAdaptiveAll(true);
    RealVector new_params_4( mkl_kernel.numberOfParameters() );
    new_params_4(0) = 1.0;
    new_params_4(2) = 0.2;
    mkl_kernel.setParameterVector( new_params_4 );
    //###end<mkl_change_something_one>

    //###begin<mkl_introspection_two>
    // examine effects
    cout << "mkl_kernel.isAdaptive(0): " << mkl_kernel.isAdaptive(0) << endl;
    cout << "mkl_kernel.isAdaptive(1): " << mkl_kernel.isAdaptive(1) << endl;
    cout << "mkl_kernel.isAdaptive(2): " << mkl_kernel.isAdaptive(2) << endl;
    cout << "mkl_kernel.numberOfParameters(): " << mkl_kernel.numberOfParameters() << endl;
    cout << "mkl_kernel.parameterVector(): " << mkl_kernel.parameterVector() << endl;
    cout << "mkl_kernel.eval( dataset.element(0), dataset.element(1) ): " << mkl_kernel.eval( dataset.element(0), dataset.element(1) ) << endl << endl;
    //###end<mkl_introspection_two>


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

    //###begin<normalization_data_setup>
    std::size_t num_dims = 9;
    std::size_t num_points = 200;
    std::vector<RealVector> input(num_points);
    RealVector v(num_dims);
    for ( std::size_t i=0; i<num_points; i++ ) {
        for ( std::size_t j=0; j<num_dims; j++ )
            v(j) = Rng::uni(-1,1);
        input[i] = v;
    }
    UnlabeledData<RealVector> rand_data = createDataFromRange( input );
    //###end<normalization_data_setup>


    //###begin<normalization_kernel_setup>
    // declare kernels
    DenseRbfKernel         unnormalized_kernel1(0.1);
    DenseLinearKernel      unnormalized_kernel2;
    DensePolynomialKernel  unnormalized_kernel3(2, 1.0);
    // declare indices
    std::vector< std::pair< std::size_t, std::size_t > > indices;
    indices.push_back( std::make_pair( 0,3 ) );
    indices.push_back( std::make_pair( 3,6 ) );
    indices.push_back( std::make_pair( 6,9 ) );
    //###end<normalization_kernel_setup>

    //###begin<normalization_trainer_setup>
    DenseScaledKernel scale( &unnormalized_kernel3 );
    NormalizeKernelUnitVariance<> normalizer;
    normalizer.train( scale, rand_data );
    //###end<normalization_trainer_setup>


    //###begin<normalization_trainer_introspect_one>
    std::cout << endl << " ======================= Kernel normalization: ======================= " << std::endl;

    std::cout << endl << "Done training. Factor is " << scale.factor() << std::endl;
    std::cout << "Mean                   = " << normalizer.mean() << std::endl;
    std::cout << "Trace                  = " << normalizer.trace() << std::endl << std::endl;
    //check in feature space
    double control = 0.0;
    for ( std::size_t i=0; i<num_points; i++ ) {
        control += scale.eval(input[i], input[i]);
        for ( std::size_t j=0; j<num_points; j++ ) {
            control -= scale.eval(input[i],input[j]) / num_points;
        }
    }
    control /= num_points;
    std::cout << "Resulting variance of scaled Kernel: " << control << std::endl << std::endl;
    //###end<normalization_trainer_introspect_one>

    //###begin<normalization_create_subrange_kernel>
    std::vector<AbstractKernelFunction<RealVector>* > kernels4;
    kernels4.push_back( &unnormalized_kernel1 );
    kernels4.push_back( &unnormalized_kernel2 );
    kernels4.push_back( &scale );
    DenseSubrangeKernel kernel4( kernels4, indices );
    //###end<normalization_create_subrange_kernel>
}
