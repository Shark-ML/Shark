#define BOOST_TEST_MODULE BLAS_Conv2d
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/mpl/list.hpp>

#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/kernels/conv2d.hpp>

using namespace shark;
using namespace blas;

template<class E1, class E2, class M>
void conv2dTest(
	blas::matrix_expression<E1, blas::cpu_tag> const& image,
	blas::matrix_expression<E2, blas::cpu_tag> const& filter,
	blas::matrix_expression<M, blas::cpu_tag>& output,
	std::size_t num_channels,
	std::size_t num_filters
){
	std::size_t filter_size1 = filter().size1()/(num_channels * num_filters);
	std::size_t filter_size2 = filter().size2();
	std::size_t output_size1 = output().size1() / num_filters;
	std::size_t output_size2 = output().size2();
	std::size_t image_size1 = image().size1()/num_channels;
	
	output().clear();
	
	for(std::size_t f = 0; f != num_filters; ++f){
		std::size_t start_out1 = f * output_size1;
		for(std::size_t c = 0; c != num_channels; ++c){
			std::size_t start_filter1 = (f * num_channels + c) * filter_size1;
			std::size_t start_image1 = c * image_size1;			
			for(std::size_t i = 0; i != output_size1; ++i){
				for(std::size_t j = 0; j != output_size2; ++j){
					for(std::size_t i0 = 0; i0 != filter_size1; ++i0){
						for(std::size_t j0 = 0; j0 != filter_size2; ++j0){
							output()(start_out1 + i,j) += image()(start_image1 + i + i0,j + j0) * filter()(start_filter1 + i0, j0); 
						}
					}
				}
			}
		}
	}
}


template<class T>
void test(
	std::size_t image_size1, std::size_t image_size2,
	std::size_t filter_size1, std::size_t filter_size2,
	std::size_t num_channels,
	std::size_t num_filters
){
	
	blas::matrix<T> image(num_channels * image_size1 , image_size2);
	blas::matrix<T> filter(num_channels * num_filters *  filter_size1, filter_size2);
	
	for(std::size_t i = 0; i != num_channels * image_size1; ++i){
		for(std::size_t j = 0; j != image_size2; ++j){
			image(i,j)  = 1.0/(num_channels * image_size1)*i + 0.1 - (0.1/image_size2)*j;
		}
	}
	for(std::size_t i = 0; i != num_channels * num_filters * filter_size1; ++i){
		for(std::size_t j = 0; j != filter_size2; ++j){
			filter(i,j)  = 1.0/(num_channels * filter_size1)*i + 0.1 - (0.1/filter_size2)*j;
		}
	}
	std::size_t output_size1 = image_size1 - filter_size1 +1;
	std::size_t output_size2 = image_size2 - filter_size2 +1;
	
	blas::matrix<T> out(output_size1 * num_filters, output_size2 ,0.0);
	blas::matrix<T> outTest(output_size1 * num_filters, output_size2 ,0.0);
	
	kernels::conv2d(image,filter,out,num_channels, num_filters);
	conv2dTest(image,filter,outTest,num_channels, num_filters);
	
	for(std::size_t i = 0; i != output_size1; ++i){
		for(std::size_t j = 0; j != output_size2; ++j){
			for(std::size_t k = 0; k != num_filters; ++k){
				double val = out(k * output_size1 + i,j);
				double valTest = outTest(k * output_size1 + i,j);
				BOOST_CHECK_CLOSE(val,valTest,1.e-3);
			}
		}
	}
}

BOOST_AUTO_TEST_SUITE(BLAS_Conv2d)



typedef boost::mpl::list<float,double> data_types;
BOOST_AUTO_TEST_CASE_TEMPLATE(syrk_test, value_type,data_types) {
	test<value_type>(32,16,4,8,5,7);
	test<value_type>(57,33,7,3,22,15);
	test<value_type>(192,333,7,3,22,15);
	test<value_type>(381,333,7,3,22,15);
}

BOOST_AUTO_TEST_SUITE_END()
