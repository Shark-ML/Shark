#ifndef UNSUPERVISED_RBM_PROBLEMS_BARSANDSTRIPES_H
#define UNSUPERVISED_RBM_PROBLEMS_BARSANDSTRIPES_H

#include  <shark/Data/Dataset.h>
#include  <shark/LinAlg/Base.h>
namespace shark{

///Generates the Bars-And-Stripes problem. In this problem, a 4x4 image has either rows or columns of the same value.
class BarsAndStripes{
private:
	UnlabeledData<RealVector> m_data;
public:
	BarsAndStripes(std::size_t batchSize = 32, bool bipolar = false){
		std::vector<RealVector> data(32,RealVector(16));
		RealVector line(4);
		for(size_t x=0; x != 16; x++) {
			for(size_t j=0; j != 4; j++) {
				line(j) = (x & (1<<j)) > 0;
				if(bipolar && line(j)==0) line(j) = -1; 
			}

			for(int i=0; i != 4; i++) {
				subrange(data[x],i*4 ,i*4 + 4) = line;
			}
			for(int i=0; i != 4; i++) {
				for(int l=0; l<4; l++) {
					data[16+x](l*4 + i) = line(l);
				}
			}
		}
		m_data = createDataFromRange(data,batchSize);
	}
	///Returns all input pattern of the BarsAndStripes problem
	UnlabeledData<RealVector> data() const{
		return m_data;
	};

	///returns the dimensionality of the data
	std::size_t inputDimension() const {
		return 16;
	}
};

}
#endif
