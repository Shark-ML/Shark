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

using namespace shark;

void LDA::train(LinearClassifier& model, LabeledData<RealVector,unsigned int> const& dataset){
	if(dataset.empty()){
		throw SHARKEXCEPTION("[LDA::train] the dataset must not be empty");
	}
	typedef LabeledData<RealVector,unsigned int>::const_batch_reference BatchReference;
	
	std::size_t inputs = dataset.numberOfElements();
	std::size_t dim = inputDimension(dataset);
	std::size_t classes = numberOfClasses(dataset);
	
	model.setStructure(dim,classes);
	
	//required statistics
	std::vector<unsigned > num(classes,0);
	std::vector<RealVector> mean(classes, RealVector(dim));
	RealMatrix covariance(dim, dim);
	
	//set class means and covariance to 0
	for(std::size_t c=0;c!=classes;++c){
		mean[c].clear();
	}
	covariance.clear();
	
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
			noalias(mean[c])+=row(points,e);
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
		mean[c] /= num[c];
		double factor = num[c];
		factor/=inputs-classes;
		noalias(covariance)-= factor*outer_prod(mean[c],mean[c]);
	}
	

	//add regularization
	if(m_regularization>0){
		for(std::size_t i=0;i!=dim;++i)
		covariance(i,i)+=m_regularization;
	}

	//slow operation - this does the matrix inversion
	model.importCovarianceMatrix(covariance);
	for (std::size_t c=0; c<classes; c++){
		model.setClassMean(c,mean[c]);
	}
}
