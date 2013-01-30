//===========================================================================
/*!
 *  \file Converter.cpp
 *
 *  \brief Converter
 *
 *  \author T.Glasmachers
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
#include <shark/Models/Converter.h>


using namespace shark;


ThresholdConverter::ThresholdConverter(double threshold)
: m_threshold(threshold){
	this->m_name = "ThresholdConverter";
}


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
	outputs.resize(pattern.size1(),1);
	for(std::size_t i = 0; i != pattern.size1(); ++i)
		outputs(i) = (pattern(i,0) > m_threshold) ? 1 : 0;
}

////////////////////////////////////////////////////////////

ThresholdVectorConverter::ThresholdVectorConverter(double threshold)
: m_threshold(threshold){
	this->m_name = "ThresholdVectorConverter";
}


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


////////////////////////////////////////////////////////////


ArgMaxConverter::ArgMaxConverter(){ 
	this->m_name = "ArgMaxConverter"; 
}

RealVector ArgMaxConverter::parameterVector() const{ 
	return RealVector(); 
}

void ArgMaxConverter::setParameterVector(RealVector const& newParameters){
	SHARK_CHECK(newParameters.size() == 0, "[ArgMaxConverter::setParameterVector] invalid number of parameters");
}

size_t ArgMaxConverter::numberOfParameters() const{ 
	return 0; 
}

void ArgMaxConverter::eval(BatchInputType const& patterns, BatchOutputType& outputs)const{
	SHARK_CHECK(patterns.size2() > 0, "[ArgMaxConverter::eval] invalid size of base model outputs");
	outputs.resize(patterns.size1(),1);
	for(std::size_t pattern = 0; pattern != patterns.size1();++pattern){
		double best = patterns(pattern,0);
		for(std::size_t i=1; i != patterns.size2(); i++){
			if (patterns(pattern,i) > best) { 
				best = patterns(pattern,i); 
				outputs(pattern) = i; 
			}
		}
	}
}


////////////////////////////////////////////////////////////


OneHotConverter::OneHotConverter(unsigned int classes)
: m_classes(classes){ 
	this->m_name = "OneHotConverter"; 
}


RealVector OneHotConverter::parameterVector() const{ 
	return RealVector(); 
}

void OneHotConverter::setParameterVector(RealVector const& newParameters){
	SHARK_CHECK(newParameters.size() == 0, "[OneHotConverter::setParameterVector] invalid number of parameters");
}

std::size_t OneHotConverter::numberOfParameters() const{ 
	return 0;
}

void OneHotConverter::eval(BatchInputType const& input, BatchOutputType& outputs)const{
	outputs.resize(input.size(),m_classes);
	outputs.clear();
	for(std::size_t pattern = 0; pattern != input.size();++pattern){
		SHARK_CHECK(input[pattern] >= m_classes, "[OneHotConverter::eval] invalid input dimension");
		outputs(pattern,input[pattern] ) = 1;
	}
}
