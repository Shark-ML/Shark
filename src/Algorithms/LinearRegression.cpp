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
}

void LinearRegression::train(LinearModel<>& model, LabeledData<RealVector, RealVector> const& dataset){
	std::size_t inputDim = inputDimension(dataset);
	std::size_t outputDim = labelDimension(dataset);
	std::size_t numInputs = dataset.numberOfElements();
	std::size_t numBatches = dataset.numberOfBatches();

	//Let P be the matrix of points with n rows and X=(P|1). the 1 rpresents the bias weight
	//Let A = X^T X + lambda * I
	//than whe have (for lambda = 0)
	//A = ( P^T P  P^T 1)
	//       ( 1^T P  1^T1)
	RealMatrix matA(inputDim+1,inputDim+1,0.0);
	blas::Blocking<RealMatrix> Ablocks(matA,inputDim,inputDim);
	//compute A and the label matrix batchwise
	typedef LabeledData<RealVector, RealVector>::const_batch_reference BatchRef;
	for (std::size_t b=0; b != numBatches; b++){
		BatchRef batch = dataset.batch(b);
		symmRankKUpdate(trans(batch.input),Ablocks.upperLeft(),true);
		noalias(column(Ablocks.upperRight(),0))+=sumRows(batch.input);
	}
	row(Ablocks.lowerLeft(),0) = column(Ablocks.upperRight(),0);
	matA(inputDim,inputDim) = numInputs;
	//X^TX+=lambda* I
	diag(Ablocks.upperLeft())+= blas::repeat(m_regularization,inputDim);
	
	
	//we also need to compute X^T L= (P^TL, 1^T L) where L is the matrix of labels 
	RealMatrix XTL(inputDim + 1,outputDim,0.0);
	for (std::size_t b=0; b != numBatches; b++){
		BatchRef batch = dataset.batch(b);
		RealSubMatrix PTL = subrange(XTL,0,inputDim,0,outputDim);
		fast_prod(trans(batch.input),batch.label,PTL,true);
		noalias(row(XTL,inputDim))+=sumRows(batch.label);
	}	
	
	//we solve the system A Beta = X^T L
	//usually this is solved via the moore penrose inverse:
	//Beta = A^-1 T
	//but it is faster und numerically more stable, if we solve it as a symmetric system
	RealMatrix beta(inputDim+1,outputDim);
	blas::solveSymmSystem<blas::SolveAXB>(matA,beta,XTL);
	
	RealMatrix matrix = subrange(trans(beta), 0, outputDim, 0, inputDim);
	RealVector offset = row(beta,inputDim);
	
	// write parameters into the model
	model.setStructure(matrix, offset);
}
