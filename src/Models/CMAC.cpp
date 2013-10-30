//===========================================================================
/*!
 *  \file CMAC.cpp
 *
 *  \brief CMAC
 *
 *  \author O.Krause
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
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

std::vector<std::size_t> CMACMap::getIndizes(ConstRealMatrixRow const &point)const {
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
void CMACMap::setStructure(std::size_t inputs, std::size_t outputs, std::size_t numberOfTilings, std::size_t numberOfTiles, double lower, double upper,bool randomTiles){
	RealMatrix bounds(inputs,2);
	for (std::size_t dim=0; dim != inputs; ++dim) {
		bounds(dim, 0) = lower;
		bounds(dim, 1) = upper;
	}
	setStructure(inputs, outputs, numberOfTilings, numberOfTiles,bounds,randomTiles);
}
void CMACMap::setStructure(std::size_t inputs, std::size_t outputs, std::size_t numberOfTilings, std::size_t numberOfTiles, RealMatrix const& bounds, bool randomTiles){
	m_inputSize  = inputs;
	m_outputSize = outputs;
	m_tilings    = numberOfTilings;
	
	m_offset.resize(numberOfTilings, inputs);
	m_dimOffset.resize(inputs + 1);
	m_tileBounds.resize(inputs, 2);
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
	numberOfParameters *= outputs;
	m_parameters.resize(numberOfParameters);
	
	//create tilings
	m_offset.clear();
	for (unsigned tiling = 0; tiling < m_tilings; ++tiling) {
		for (unsigned dim = 0; dim < m_inputSize; ++dim) {
			if (!randomTiles)
				m_offset(tiling, dim) -= 0.5*m_tileBounds(dim,1)*(1.0+tiling)/m_tilings;
			else
				m_offset(tiling, dim) -= 0.5*Rng::uni(0, m_tileBounds(dim,1));
		}
	}
}

void CMACMap::eval(RealMatrix const& patterns,RealMatrix &output) const{
	SIZE_CHECK(patterns.size2() == m_inputSize);
	std::size_t numPatterns = patterns.size1();
	output.resize(numPatterns,m_outputSize);
	output.clear();
	//todo: fix axpy_prod for sparse matrix before trying this again...
	//create feature matrix for the batch. we assume that it is sparse and that tiles>>tilings
// 	RealCompressedMatrix features(numPatterns,m_outputSize);
// 	for(std::size_t i = 0; i != numPatterns; ++i){
// 		std::vector<std::size_t> indizes = getIndizes(pattern);
// 		for (std::size_t j = 0; j != m_tilings; ++j) {
// 			features(i,j) = 1; 
// 		}
// 	}
// 	
// 	axpy_prod(features,trans(m_matrix),output,false);
	
	for(std::size_t i = 0; i != numPatterns; ++i){
		std::vector<std::size_t> indizes = getIndizes(row(patterns,i));
		for (std::size_t o=0; o!=m_outputSize; ++o) {
			for (std::size_t j = 0; j != m_tilings; ++j) {
				output(i,o) += m_parameters(indizes[j] + o*m_parametersPerTiling);
			}
		}
	}
}

void CMACMap::weightedParameterDerivative(
	RealMatrix const& patterns, 
	RealMatrix const& coefficients, 
	State const&,//not needed
	RealVector &gradient
) const{
	SIZE_CHECK(patterns.size2() == m_inputSize);
	SIZE_CHECK(coefficients.size2() == m_outputSize);
	SIZE_CHECK(coefficients.size1() == patterns.size1());
	std::size_t numPatterns = patterns.size1();
	gradient.resize(m_parameters.size());
	gradient.clear();
	for(std::size_t i = 0; i != numPatterns; ++i){
		std::vector<std::size_t> indizes = getIndizes(row(patterns,i));
		for (std::size_t o=0; o!=m_outputSize; ++o) {
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
	archive >> m_outputSize;
	archive >> m_parameters;
}

void CMACMap::write(OutArchive &archive) const {
	archive << m_offset;
	archive << m_dimOffset;
	archive << m_tileBounds;
	archive << m_tilings;
	archive << m_parametersPerTiling;
	archive << m_inputSize;
	archive << m_outputSize;
	archive << m_parameters;
}
