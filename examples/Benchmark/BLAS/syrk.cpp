#include <shark/LinAlg/BLAS/blas.h>
#include <shark/LinAlg/BLAS/kernels/syrk.hpp>
#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

template<class Triangular, class AMat, class CMat>
double benchmark(
	blas::matrix_expression<AMat, blas::cpu_tag> const& A,
	blas::matrix_expression<CMat, blas::cpu_tag> & C
){
	double minTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 10; ++i){
		Timer time;
		shark::blas::kernels::syrk<Triangular::is_upper>(A,C,2.0);
		minTime = min(minTime,time.stop());
	}
	return (0.5*A().size1()*A().size2()*A().size1())/minTime/1024/1024;
}

int main(int argc, char **argv) {
	std::size_t size = 100;
	std::cout<<"Mega Flops"<<std::endl;
	for(std::size_t iter = 0; iter != 5; ++iter){
		blas::matrix<double,blas::row_major> Arow(size,size);
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != size; ++k){
				Arow(i,k)  = 0.1/size*i+0.1/size*k;
			}
		}
		blas::matrix<double,blas::column_major> Acol = Arow;
		
		blas::matrix<double,blas::row_major> Crow(size,size,0.0);
		blas::matrix<double,blas::column_major> Ccol(size,size,0.0);
		std::cout<<size<<"\trow major result - lower\t"<<benchmark<blas::lower>(Arow,Crow)<<"\t"<< benchmark<blas::lower>(Acol,Crow)<<std::endl;
		std::cout<<size<<"\trow major result - upper\t"<<benchmark<blas::upper>(Arow,Crow)<<"\t"<< benchmark<blas::upper>(Acol,Crow)<<std::endl;
		std::cout<<size<<"\tcolumn major result - lower\t"<<benchmark<blas::lower>(Arow,Ccol)<<"\t"<< benchmark<blas::lower>(Acol,Ccol)<<std::endl;
		std::cout<<size<<"\tcolumn major result - upper\t"<<benchmark<blas::upper>(Arow,Ccol)<<"\t"<< benchmark<blas::upper>(Acol,Ccol)<<std::endl;


		std::cout<<std::endl;
		size *=2;
	}
}