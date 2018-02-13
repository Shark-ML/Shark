//===========================================================================
/*!
 * 
 *
 * \brief       CMAC
 * 
 * 
 *
 * \author      O.Krause
 * \date        2010-2011
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
//===========================================================================
#define SHARK_COMPILE_DLL
#include <shark/Models/CMAC.h>
#include <boost/serialization/vector.hpp>
using namespace shark;

std::size_t CMACMap::getArrayIndexForTiling(std::size_t indexOfTiling,RealVector const &point)const {

	std::size_t index = indexOfTiling * m_dimOffset[m_inputSize];

	for (std::size_t dim = 0; dim != m_inputSize; ++dim) {
		//Adjust range from (lower bound, upper bound) to (0,numberOfTiles)
		double coordinate = point(dim);
		coordinate -= m_tileBounds(dim, 0);//subtract lower bound
		coordinate -= m_offset(indexOfTiling, dim);//tiling offset
		//divide by the width of the tile to calculate the index
		coordinate /= m_tileBounds(dim, 1);
		//add index offset
		index += static_cast<std::size_t>(coordinate) * m_dimOffset[dim];
	}
	return index;
}

std::vector<std::size_t> CMACMap::getIndizes(blas::dense_vector_adaptor<double const> const &point)const {
	std::vector<size_t> output(m_tilings,0);

	for (size_t tiling = 0; tiling != m_tilings; ++tiling) {
		size_t index = getArrayIndexForTiling(tiling, point);
		output[tiling] = index;
	}
	return output;
}

CMACMap::CMACMap():m_tilings(0) {
	m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
}
void CMACMap::setStructure(Shape const& inputs, Shape const& outputs, std::size_t numberOfTilings, std::size_t numberOfTiles, double lower, double upper,bool randomTiles){
	RealMatrix bounds(inputs.numElements(),2);
	for (std::size_t dim=0; dim != bounds.size1(); ++dim) {
		bounds(dim, 0) = lower;
		bounds(dim, 1) = upper;
	}
	setStructure(inputs, outputs, numberOfTilings, numberOfTiles,bounds,randomTiles);
}
void CMACMap::setStructure(Shape const& inputs, Shape const& outputs, std::size_t numberOfTilings, std::size_t numberOfTiles, RealMatrix const& bounds, bool randomTiles){
	m_inputSize  = inputs.numElements();
	m_inputShape = inputs;
	m_outputShape = outputs;
	m_tilings    = numberOfTilings;
	
	m_offset.resize(numberOfTilings, m_inputSize);
	m_dimOffset.resize(m_inputSize + 1);
	m_tileBounds.resize(m_inputSize, 2);
	m_tilings = numberOfTilings;
	
	//initialize bounds
	for (std::size_t dim=0; dim != m_inputSize; ++dim) {
		double tileWidth = (bounds(dim, 1) - bounds(dim, 0)) / (numberOfTiles - 1);
		m_tileBounds(dim, 0) = bounds(dim, 0);
		m_tileBounds(dim, 1) = tileWidth;
	}

	//calculate number of parameters and the offsets for every input dimension
	std::size_t numberOfParameters = 1;
	for (std::size_t inputDim = 0; inputDim != m_inputSize; ++inputDim) {
		m_dimOffset[inputDim] = numberOfParameters;
		numberOfParameters *= numberOfTiles;
	}
	//parameters per tiling
	m_dimOffset[m_inputSize] = numberOfParameters;

	//parameters for each output dimension
	numberOfParameters *= m_tilings;
	m_parametersPerTiling=numberOfParameters;
	//parameters total
	numberOfParameters *= outputs.numElements();
	m_parameters.resize(numberOfParameters);
	
	//create tilings
	m_offset.clear();
	for (unsigned tiling = 0; tiling < m_tilings; ++tiling) {
		for (unsigned dim = 0; dim < m_inputSize; ++dim) {
			if (!randomTiles)
				m_offset(tiling, dim) -= 0.5*m_tileBounds(dim,1)*(1.0+tiling)/m_tilings;
			else
				m_offset(tiling, dim) -= 0.5*random::uni(random::globalRng, std::size_t(0), m_tileBounds(dim,1));
		}
	}
}

void CMACMap::eval(RealMatrix const& patterns,RealMatrix &output) const{
	SIZE_CHECK(patterns.size2() == m_inputSize);
	std::size_t numPatterns = patterns.size1();
	output.resize(numPatterns, m_outputShape.numElements());
	output.clear();
	
	for(std::size_t i = 0; i != numPatterns; ++i){
		auto indizes = getIndizes(row(patterns,i));
		for (std::size_t o = 0; o != output.size2(); ++o) {
			for (std::size_t j = 0; j != m_tilings; ++j) {
				output(i,o) += m_parameters(indizes[j] + o*m_parametersPerTiling);
			}
		}
	}
}

void CMACMap::weightedParameterDerivative(
	RealMatrix const& patterns, 
	BatchOutputType const& outputs,
	RealMatrix const& coefficients, 
	State const&,//not needed
	RealVector &gradient
) const{
	SIZE_CHECK(patterns.size2() == m_inputSize);
	SIZE_CHECK(coefficients.size2() == outputs.size2());
	SIZE_CHECK(coefficients.size1() == patterns.size1());
	std::size_t numPatterns = patterns.size1();
	gradient.resize(m_parameters.size());
	gradient.clear();
	for(std::size_t i = 0; i != numPatterns; ++i){
		auto indizes = getIndizes(row(patterns,i));
		for (std::size_t o=0; o!= outputs.size2(); ++o) {
			for (std::size_t j=0; j != m_tilings; ++j) {
				gradient(indizes[j] + o*m_parametersPerTiling) += coefficients(i,o);
			}
		}
	}
}


void CMACMap::read(InArchive &archive) {
	archive >> m_offset;
	archive >> m_dimOffset;
	archive >> m_tileBounds;
	archive >> m_tilings;
	archive >> m_parametersPerTiling;
	archive >> m_inputSize;
	archive >> m_inputShape;
	archive >> m_outputShape;
	archive >> m_parameters;
}

void CMACMap::write(OutArchive &archive) const {
	archive << m_offset;
	archive << m_dimOffset;
	archive << m_tileBounds;
	archive << m_tilings;
	archive << m_parametersPerTiling;
	archive << m_inputSize;
	archive << m_inputShape;
	archive << m_outputShape;
	archive << m_parameters;
}
