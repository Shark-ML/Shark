#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/kernels/potrf.hpp>
#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

template<class MatA, class Triang>
double benchmark(
	blas::matrix_expression<MatA, blas::cpu_tag> const& A,
	Triang
){
	double minTime = std::numeric_limits<double>::max();
	volatile double res = 0;
	for(std::size_t i = 0; i != 20; ++i){
		typename blas::matrix_temporary<MatA>::type Acopy =A;
		Timer time;
		blas::kernels::potrf<Triang>(Acopy);
		minTime = min(minTime,time.stop());
		res += max(Acopy);
	}
	return (1.0/3.0*A().size1()*A().size1()*A().size1())/minTime/1024/1024;
}

int main(int argc, char **argv) {
	std::size_t size = 128;
	std::cout<<"Mega Flops"<<std::endl;
	for(std::size_t iter = 0; iter != 10; ++iter){
		blas::matrix<double,blas::row_major> Arow(size,size);
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t j = 0; j != size; ++j){
				Arow(i,j)  = 0.1/size*i+0.1/size*j;
			}
			Arow(i,i) += 1000.0;
		}
		blas::matrix<double,blas::column_major> Acol = Arow;
		std::cout<<size<<"\t upper\t"<<benchmark(Arow,blas::upper())<<"\t"<< benchmark(Acol,blas::upper())<<std::endl;
		std::cout<<size<<"\t lower\t"<<benchmark(Arow,blas::lower())<<"\t"<< benchmark(Acol,blas::lower())<<std::endl;
		size *=2;
	}
}