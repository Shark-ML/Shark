extern "C"{
#include <cblas.h>
}
int main(){
	float* x;
	float* y;
	int N,strideX,strideY;
	float r= cblas_sdot(N, x, strideX, y, strideY);
}