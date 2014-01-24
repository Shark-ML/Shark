//===========================================================================
/*!
 * 
 * \file        Converter.cpp
 *
 * \brief       Converter
 * 
 * 
 *
 * \author      T.Glasmachers
 * \date        2010-2011
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
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
#include <shark/Models/Converter.h>


using namespace shark;


ThresholdConverter::ThresholdConverter(double threshold)
: m_threshold(threshold)
{ }


RealVector ThresholdConverter::parameterVector() const{ 
	return RealVector(); 
}

void ThresholdConverter::setParameterVector(RealVector const& newParameters){
	SHARK_CHECK(newParameters.size() == 0, "[ThresholdConverter::setParameterVector] invalid number of parameters");
}

std::size_t ThresholdConverter::numberOfParameters() const{ 
	return 0; 
}

void ThresholdConverter::eval(BatchInputType const& pattern, BatchOutputType& outputs)const{
	SHARK_CHECK(pattern.size2() == 1, "[ThresholdConverter::eval] invalid size of input");
	outputs.resize(pattern.size1());
	for(std::size_t i = 0; i != pattern.size1(); ++i)
		outputs(i) = (pattern(i,0) > m_threshold) ? 1 : 0;
}

////////////////////////////////////////////////////////////

ThresholdVectorConverter::ThresholdVectorConverter(double threshold)
: m_threshold(threshold)
{ }


RealVector ThresholdVectorConverter::parameterVector() const{
	return RealVector(); 
}

void ThresholdVectorConverter::setParameterVector(RealVector const& newParameters){
	SHARK_CHECK(newParameters.size() == 0, "[ThresholdVectorConverter::setParameterVector] invalid number of parameters");
}

std::size_t ThresholdVectorConverter::numberOfParameters() const{ 
	return 0; 
}

void ThresholdVectorConverter::eval(BatchInputType const& patterns, BatchOutputType& outputs)const{
	outputs.resize(patterns.size1(),patterns.size2());
	for(std::size_t i = 0; i != patterns.size1(); ++i){
		for(std::size_t j = 0; j != patterns.size2(); ++j){
			outputs(i,j) = patterns(i,j) > m_threshold;
		}
	}
}
