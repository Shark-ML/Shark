/*!
 * \brief       Loads the MNIST benchmark problem.
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
#ifndef UNSUPERVISED_RBM_PROBLEMS_MNIST_H
#define UNSUPERVISED_RBM_PROBLEMS_MNIST_H

#include <shark/Data/Dataset.h>
#include <shark/LinAlg/Base.h>
#include <shark/Core/Random.h>

#include <sstream>
#include <fstream>
#include <string>
namespace shark{

/// \brief Reads in the famous MNIST data in possibly binarized form. The MNIST database itself is not included in Shark, 
///  this class just helps loading it.
///
///MNIST is a set of handwritten digits.
///It needs the filename of the file containing the database (can be downloaded form the web)
///and the threshold for binarization. The threshold (between 0 and 255) describes when a gray value will be interpreted
///as 1. Default is 127. If the threshold is 0, no binarization takes place.
class MNIST{
private:
	UnlabeledData<RealVector> m_data;
	std::string m_filename;
	char m_threshold;
	std::size_t m_batchSize;

	int readInt (unsigned char *memblock) const{
		return ((int)memblock[0] << 24) + ((int)memblock[1] << 16) + ((int)memblock[2] << 8) + memblock[3];
	}
	void init(){
		//m_name="MNIST";
		std::ifstream infile(m_filename.c_str(), std::ios::binary);
		SHARK_RUNTIME_CHECK(infile, "Can not open file!");
		
		//get file size
		infile.seekg(0,std::ios::end);
		std::ifstream::pos_type inputSize = infile.tellg();
		
		
		unsigned char *memblock = new unsigned char [inputSize];
		infile.seekg (0, std::ios::beg);
		infile.read ((char *) memblock, inputSize);
        
		SHARK_RUNTIME_CHECK(readInt(memblock) == 2051, "magic number for mnist wrong!");
		std::size_t numImages = readInt(memblock + 4);
		std::size_t numRows = readInt(memblock + 8);
		std::size_t numColumns = readInt(memblock + 12);
		std::size_t sizeOfVis = numRows * numColumns;
		
		std::vector<RealVector> data(numImages,RealVector(sizeOfVis));
		for (std::size_t i = 0; i != numImages; ++i){
			RealVector imgVec(sizeOfVis);
			if(m_threshold  != 0){ 
				for (size_t j = 0; j != sizeOfVis; ++j){
					char pixel = memblock[ 16 + i * sizeOfVis + j ] > m_threshold;
					data[i](j) = pixel;
				}
			}
			else{
				for (size_t j = 0; j != sizeOfVis; ++j){
					data[i](j) = memblock[ 16 + i * sizeOfVis + j ];
				}
			}
		}
		delete [] memblock;
		m_data = createDataFromRange(data,m_batchSize);
	}
public:
	
	//Constructor. Sets the configurations from a property tree and imports the data set.
	//@param filename the name of the file storing the dataset
	//@param threshhold the threshold for turning gray values into ones
	//@param batchSize the size of the batch 
	MNIST(std::string filename, char threshold = 127, std::size_t batchSize = 256)
	        :  m_filename(filename), m_threshold(threshold), m_batchSize(batchSize){
		init();
	}
	
	//Returns the data vector
	UnlabeledData<RealVector> data() const {
		return m_data;
	}	
	
	//Returns the dimension of the pattern of MNIST.
	std::size_t inputDimension() const {
		return 28*28;
	}

	//Returns the batch size.
	std::size_t batchSize() const {
		return m_batchSize;
	}

};
}
#endif

