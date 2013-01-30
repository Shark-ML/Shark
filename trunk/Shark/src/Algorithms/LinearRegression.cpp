//===========================================================================
/*!
 *  \file LinearRegression.cpp
 *
 *  \brief LinearRegression
 *
 *  \author O.Krause, T. Glasmachers
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
#include <shark/LinAlg/solveSystem.h>
#include <shark/Algorithms/Trainers/LinearRegression.h>

using namespace shark;


LinearRegression::LinearRegression(double regularization){
	setRegularization(regularization);
	m_name = "Linear Regression";
}

void LinearRegression::train(LinearModel<>& model, LabeledData<RealVector, RealVector> const& dataset){
	std::size_t inputDim = inputDimension(dataset);
	std::size_t outputDim = labelDimension(dataset);
	std::size_t numInputs = dataset.numberOfElements();

	// copy the input and target data into matrix format
	RealMatrix X(numInputs, inputDim + 1);
	RealMatrix targetMatrix(outputDim, numInputs);
	targetMatrix.clear();
	LabeledData<RealVector, RealVector>::const_element_iterator elem=dataset.elemBegin();
	for (std::size_t e=0; e != numInputs; e++){
 		for (std::size_t i=0; i != inputDim; i++){
 			X(e, i) = elem->input(i);
 		}
		X(e, inputDim) = 1.0;
		column(targetMatrix, e) = elem->label;
		++elem;
	}
	
	//Let A = X^T X + lambda * I
	//we solve the system A Beta = X^T T
	//usually this is solved via the moore penrose inverse:
	//Beta = A^-1 X^T T
	//but it is faster und numerically more stable, if we solve it as a symmetric system

	// calculation of (X^T X + lambda * I)^-1 * X^T
	RealMatrix matA(inputDim+1,inputDim+1);
	symmRankKUpdate(trans(X),matA);//A=X^T X 
	//A+=lambda * I
	if(m_regularization){
		for (std::size_t i=0; i != inputDim; ++i) {
			matA(i, i) += m_regularization;
		}
	}
	//calculate X^T T
	RealMatrix XTT(inputDim + 1,outputDim);
	fast_prod(trans(X),trans(targetMatrix),XTT);
	X.resize(0,0);//save memory, X not needed anymore
	
	RealMatrix beta(inputDim+1,outputDim);
	solveSymmSystem<SolveAXB>(matA,beta,XTT);
	
	RealMatrix matrix = subrange(trans(beta), 0, outputDim, 0, inputDim);
	RealVector offset = row(beta,inputDim);
	
	// write parameters into the model
	model.setStructure(matrix, offset);
}
