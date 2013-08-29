//===========================================================================
/*!
 *  \file LinearNorm.cpp
 *
 *  \brief LinearNorm
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
#include <shark/Models/LinearNorm.h>
using namespace shark;

LinearNorm::LinearNorm():m_inputSize(0)
{
	m_features|=HAS_FIRST_INPUT_DERIVATIVE;
	m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
}
LinearNorm::LinearNorm(std::size_t inputSize):m_inputSize(inputSize)
{
	m_features|=HAS_FIRST_INPUT_DERIVATIVE;
	m_features|=HAS_FIRST_PARAMETER_DERIVATIVE;
}

//! configures the network
//!
//!This class only needs one item to be presents, the number of inputs "input"
void LinearNorm::configure( PropertyTree const& node ){
	m_inputSize = node.get<std::size_t>("inputs");
}

void LinearNorm::eval(BatchInputType const& patterns, BatchOutputType& output)const{
	SIZE_CHECK(patterns.size2() == inputSize());
	output.resize(patterns.size1(),inputSize());
	for(std::size_t i = 0; i != patterns.size1(); ++i){
		double norm =sum(row(patterns,i));
		noalias(row(output,i))=row(patterns,i)/norm;
	}
}
void LinearNorm::eval(BatchInputType const& patterns, BatchOutputType& output, State& state)const{
	SIZE_CHECK(patterns.size2() == inputSize());
	output.resize(patterns.size1(),inputSize());
	InternalState& s = state.toState<InternalState>();
	s.resize(patterns.size1());
	for(std::size_t i = 0; i != patterns.size1(); ++i){
		s.norm(i)=sum(row(patterns,i));
		noalias(row(output,i))=row(patterns,i)/s.norm(i);
	}
}

void LinearNorm::weightedInputDerivative(
	BatchInputType const& patterns, BatchOutputType const& coefficients, State const& state, BatchOutputType& gradient
)const{
	SIZE_CHECK(patterns.size2() == inputSize());
	SIZE_CHECK(coefficients.size1()==patterns.size1());
	SIZE_CHECK(coefficients.size2()==patterns.size2());
	gradient.resize(patterns.size1(),inputSize());
	InternalState const& s = state.toState<InternalState>();
	for(std::size_t i = 0; i != patterns.size1(); ++i){
		double norm_sqr=sqr(s.norm(i));
		double constant=inner_prod(row(coefficients,i),row(patterns,i))/norm_sqr;
		noalias(row(gradient,i))=row(coefficients,i)/s.norm(i)- blas::repeat(constant,inputSize());
	}
}


void LinearNorm::read( InArchive & archive ){
	archive >> m_inputSize;
}


void LinearNorm::write( OutArchive & archive ) const{
	archive << m_inputSize;
}
