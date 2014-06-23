//===========================================================================
/*!
 * 
 *
 * \brief       LDA
 * 
 * 
 *
 * \author      O.Krause
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
#include <shark/Algorithms/Trainers/LDA.h>
#include <shark/LinAlg/solveSystem.h>

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
	UIntVector num(classes,0);
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
			++num(c);
			noalias(row(means,c))+=row(points,e);
		}
		//update second moment matrix
		symm_prod(trans(points),covariance,false);
	}
	covariance/=inputs-classes;
	//calculate mean and the covariance matrix from second moment
	for (std::size_t c = 0; c != classes; c++){
		if (num[c] == 0) 
			throw SHARKEXCEPTION("[LDA::train] LDA can not handle a class without examples");
		row(means,c) /= num(c);
		double factor = num(c);
		factor/=inputs-classes;
		noalias(covariance)-= factor*outer_prod(row(means,c),row(means,c));
	}
	

	//add regularization
	if(m_regularization>0){
		for(std::size_t i=0;i!=dim;++i)
			covariance(i,i)+=m_regularization;
	}
	
	//the formula for the linear classifier is
	// arg max_i log(P(x|i) * P(i))
	//= arg max_i log(P(x|i)) +log(P(i))
	//= arg max_i -(x-m_i)^T C^-1 (x-m_i) +log(P(i))
	//= arg max_i -m_i^T C^-1 m_i  +2* x^T C^-1 m_i + log(P(i))
	//so we compute first C^-1 m_i and then the first term
	
	// compute z = m_i^T C^-1  <=>  z C = m_i 
	// this is the expensive step of the calculation.
	RealMatrix transformedMeans = means;
	blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(covariance,transformedMeans);
	
	//compute bias terms m_i^T C^-1 m_i - log(P(i))
	RealVector bias(classes);
	for(std::size_t c = 0; c != classes; ++c){
		double prior = std::log(double(num(c))/inputs);
		bias(c) = - 0.5* inner_prod(row(means,c),row(transformedMeans,c)) + prior;
	}

	//fill the model
	model.decisionFunction().setStructure(transformedMeans,bias);
}

void LDA::train(LinearClassifier<>& model, WeightedLabeledData<RealVector,unsigned int> const& dataset){
	if(dataset.empty()){
		throw SHARKEXCEPTION("[LDA::train] the dataset must not be empty");
	}
	typedef WeightedLabeledData<RealVector,unsigned int>::const_batch_reference BatchReference;
	
	std::size_t inputs = dataset.numberOfElements();
	std::size_t dim = inputDimension(dataset);
	std::size_t classes = numberOfClasses(dataset);

	//required statistics
	RealMatrix means(classes, dim,0.0);
	RealMatrix covariance(dim, dim,0.0);
	double weightSum = sumOfWeights(dataset);
	RealVector classWeight(classes,0.0);
	
	//we compute the data batch wise
	BOOST_FOREACH(BatchReference batch, dataset.batches()){
		UIntVector const& labels = batch.data.label;
		RealMatrix points = batch.data.input;
		RealVector const& weights = batch.weight;
		//load batch and update mean
		std::size_t currentBatchSize = points.size1();
		for (std::size_t e=0; e != currentBatchSize; e++){
			//update mean and class count for this sample
			std::size_t c = labels(e);
			classWeight(c) += weights(e);
			noalias(row(means,c)) += weights(e)*row(points,e);
			row(points,e) *= std::sqrt(weights(e));
			
		}
		//update second moment matrix
		symm_prod(trans(points),covariance,false);
	}
	covariance /= weightSum;
	
	//calculate mean and the covariance matrix from second moment
	for (std::size_t c = 0; c != classes; c++){
		if (classWeight[c] == 0.0) 
			throw SHARKEXCEPTION("[LDA::train] LDA can not handle a class without examples");
		row(means,c) /= classWeight(c);
		double factor = classWeight(c) / weightSum;
		noalias(covariance)-= factor*outer_prod(row(means,c),row(means,c));
	}
	

	//add regularization
	noalias(diag(covariance)) += blas::repeat(m_regularization,dim);
	
	//the formula for the linear classifier is
	// arg max_i log(P(x|i) * P(i))
	//= arg max_i log(P(x|i)) +log(P(i))
	//= arg max_i -(x-m_i)^T C^-1 (x-m_i) +log(P(i))
	//= arg max_i -m_i^T C^-1 m_i  +2* x^T C^-1 m_i + log(P(i))
	//so we compute first C^-1 m_i and then the first term
	
	// compute z = m_i^T C^-1  <=>  z C = m_i 
	// this is the expensive step of the calculation.
	RealMatrix transformedMeans = means;
	blas::solveSymmSemiDefiniteSystemInPlace<blas::SolveXAB>(covariance,transformedMeans);
	
	//compute bias terms m_i^T C^-1 m_i - log(P(i))
	RealVector bias(classes);
	for(std::size_t c = 0; c != classes; ++c){
		double prior = std::log(classWeight(c)/weightSum);
		bias(c) = - 0.5* inner_prod(row(means,c),row(transformedMeans,c)) + prior;
	}

	//fill the model
	model.decisionFunction().setStructure(transformedMeans,bias);
}
