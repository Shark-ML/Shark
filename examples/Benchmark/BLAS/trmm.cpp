#include <shark/LinAlg/BLAS/blas.h>
#include <shark/Core/Timer.h>
#include <iostream>
using namespace shark;
using namespace std;

template<class Triangular, class AMat, class BMat, class CMat>
double benchmark(
	blas::matrix_expression<AMat, blas::cpu_tag> const& A,
	blas::matrix_expression<BMat, blas::cpu_tag> const& B,
	blas::matrix_expression<CMat, blas::cpu_tag> & C
){
	double minTime = std::numeric_limits<double>::max();
	for(std::size_t i = 0; i != 10; ++i){
		Timer time;
		noalias(C) += blas::triangular_prod<Triangular>(A,B);
		minTime = min(minTime,time.stop());
	}
	return (0.5*A().size1()*A().size2()*B().size2())/minTime/1024/1024;
}

int main(int argc, char **argv) {
	std::size_t size = 100;
	std::cout<<"Mega Flops"<<std::endl;
	for(std::size_t iter = 0; iter != 10; ++iter){
		blas::matrix<double,blas::row_major> Arow(size,size);
		for(std::size_t i = 0; i != size; ++i){
			for(std::size_t k = 0; k != size; ++k){
				Arow(i,k)  = 0.1/size*i+0.1/size*k;
			}
		}
		
		blas::matrix<double,blas::row_major> Brow(size,size);
		for(std::size_t k = 0; k != size; ++k){
			for(std::size_t j = 0; j != size; ++j){
				Brow(k,j) = 0.1/size*j+0.1/size*k;
			}
		}
		blas::matrix<double,blas::column_major> Acol = Arow;
		blas::matrix<double,blas::column_major> Bcol = Brow;
		
		blas::matrix<double,blas::row_major> Crow(size,size,0.0);
		blas::matrix<double,blas::column_major> Ccol(size,size,0.0);
		std::cout<<size<<"\t row major result - lower\t"<<benchmark<blas::lower>(Arow,Brow,Crow)<<"\t"<< benchmark<blas::lower>(Acol,Brow,Crow)
		<<"\t"<< benchmark<blas::lower>(Arow,Bcol,Crow) <<"\t" <<benchmark<blas::lower>(Acol,Bcol,Crow) <<std::endl;
		std::cout<<size<<"\t row major result - upper\t"<<benchmark<blas::upper>(Arow,Brow,Crow)<<"\t"<< benchmark<blas::upper>(Acol,Brow,Crow)
		<<"\t"<< benchmark<blas::upper>(Arow,Bcol,Crow) <<"\t" <<benchmark<blas::upper>(Acol,Bcol,Crow) <<std::endl;
		std::cout<<size<<"\t column major result - lower\t"<<benchmark<blas::lower>(Arow,Brow,Ccol)<<"\t"<< benchmark<blas::lower>(Acol,Brow,Ccol)
		<<"\t"<< benchmark<blas::lower>(Arow,Bcol,Ccol) <<"\t" <<benchmark<blas::lower>(Acol,Bcol,Ccol) <<std::endl;
		std::cout<<size<<"\t column major result - upper\t"<<benchmark<blas::upper>(Arow,Brow,Ccol)<<"\t"<< benchmark<blas::upper>(Acol,Brow,Ccol)
		<<"\t"<< benchmark<blas::upper>(Arow,Bcol,Ccol) <<"\t" <<benchmark<blas::upper>(Acol,Bcol,Ccol) <<std::endl;
		std::cout<<std::endl;
		size *=2;
	}
}