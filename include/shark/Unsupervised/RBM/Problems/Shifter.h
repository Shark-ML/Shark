/*!
 * \brief       Implements the Shifter benchmark problem.
 * 
 * \author      O. Krause, A.Fischer, K.Bruegge
 * \date        2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#ifndef UNSUPERVISED_RBM_PROBLEMS_SHIFTER_H
#define UNSUPERVISED_RBM_PROBLEMS_SHIFTER_H

#include  <shark/Data/Dataset.h>
#include  <shark/LinAlg/Base.h>


namespace shark{

///Shifter problem
class Shifter{
private:
	UnlabeledData<RealVector> m_data;
public:
	Shifter(){
		std::vector<RealVector> data(768,RealVector(19));
		for(unsigned  x=0; x<=255; x++) {
			RealVector element(19);
			for(size_t i=0; i<8; i++) {
				element(i) = (x & (1<<i)) > 0;
			}
			for(int label=0; label<=2; label++) {
				unsigned char y;
				if(label==0) {
					y = (x<<1 | x>>7);
					element(16)=1;
					element(17)=0;
					element(18)=0;
				}
				else if(label==1) {	
					y = x;
					element(16)=0;
					element(17)=1;
					element(18)=0;
				}
				else {
					y = (x>>1 | x<<7);
					element(16)=0;
					element(17)=0;
					element(18)=1;
				}
				for(size_t i=0; i<8; i++) {
					element(i+8) = (y & (1<<i)) > 0;
				}
				data[x*3+label]=element;
			}
		}
		m_data = createDataFromRange(data);
	}
	
	///returns the generated dataset
	UnlabeledData<RealVector> data() const{
		return m_data;
	};
	///returns the dimensionality of the data
	std::size_t inputDimension() const {
		return 19;
	}
};
}
#endif
