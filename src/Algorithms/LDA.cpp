//===========================================================================
/*!
 *  \file LDA.cpp
 *
 *  \brief LDA
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
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/LinAlg/Inverse.h>

using namespace shark;

void LDA::train(LinearClassifier<>& model, LabeledData<RealVector,unsigned int> const& dataset){
	if(dataset.empty()){
		throw SHARKEXCEPTION("[LDA::train] the dataset must not be empty");
	}
	typedef LabeledData<RealVector,unsigned int>::const_batch_reference BatchReference;
	
	std::size_t inputs = dataset.numberOfElements();
	std::size_t dim = inputDimension(dataset);
	std::size_t classes = numberOfClasses(dataset);

	//required statistics
	std::vector<unsigned > num(classes,0);
	RealMatrix means(classes, dim,0.0);
	RealMatrix covariance(dim, dim,0.0);
	
	//we compute the data batch wise
	BOOST_FOREACH(BatchReference batch, dataset.batches()){
		UIntVector const& labels = batch.label;
		RealMatrix const& points = batch.input;
		//load batch and update mean
		std::size_t currentBatchSize = points.size1();
		for (std::size_t e=0; e != currentBatchSize; e++){
			//update mean and class count for this sample
			std::size_t c = labels(e);
			++num[c];
			noalias(row(means,c))+=row(points,e);
		}
		//update second moment matrix
		//fast_prod(trans(batch),batch,covariance,1.0);
		symmRankKUpdate(trans(points),covariance,1.0);
	}
	covariance/=inputs-classes;
	//calculate mean and the covariance matrix from second moment
	for (std::size_t c = 0; c != classes; c++){
		if (num[c] == 0) 
			throw SHARKEXCEPTION("[LDA::train] LDA can not handle a class without examples");
		row(means,c) /= num[c];
		double factor = num[c];
		factor/=inputs-classes;
		noalias(covariance)-= factor*outer_prod(row(means,c),row(means,c));
	}
	

	//add regularization
	if(m_regularization>0){
		for(std::size_t i=0;i!=dim;++i)
			covariance(i,i)+=m_regularization;
	}
	
	//the formula for the linear classifier is
	//arg min_i (x-m_i)^T C^-1 (x-m_i)
	//which is equivalent to
	//arg min_i m_i^T C^-1 m_i  -2* x^T C^-1 m_i
	//arg max_i -m_i^T C^-1 m_i  +2* x^T C^-1 m_i
	//so we compute first C^-1 m_i and than the first term
	
	//invert the matrix, take into account that it is not necessarily positive definite
	RealMatrix CInverse;
	decomposedGeneralInverse(covariance, CInverse);
	
	//multiply the mean with the inverse matrix
	RealMatrix transformedMeans(classes,dim);
	fast_prod(means,CInverse, transformedMeans);
	transformedMeans*=-1;//transform to maximisation problem
	
	//compute bias terms m_i^T C^-1 m_i
	RealVector bias(classes);
	for(std::size_t i = 0; i != classes; ++i){
		bias(i) = inner_prod(row(means,i),row(transformedMeans,i));
	}
	transformedMeans *= -2.0;

	//fill the model
	model.decisionFunction().setStructure(transformedMeans,bias);
}
