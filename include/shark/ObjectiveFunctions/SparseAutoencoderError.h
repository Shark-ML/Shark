/*!
 * 
 *
 * \brief       Specific error function for Feed-Forward-Networks which enforces it to have sparse hidden neuron activation
 * 
 * 
 *
 * \author      O.Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_SparseAutoencoderError_H
#define SHARK_OBJECTIVEFUNCTIONS_SparseAutoencoderError_H


#include <shark/Models/Autoencoder.h>
#include <shark/Models/TiedAutoencoder.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include "Impl/FunctionWrapperBase.h"

#include <boost/scoped_ptr.hpp>
namespace shark{

///
/// \brief Error Function for Autoencoders and TiedAutoencoders which should be trained with sparse activation of the hidden neurons
///
/// This error function optimizes a Network with respect to some loss function similar to the standard
/// ErrorFunction. Additionally another penalty term is added which enforces a sparse activation pattern of
/// the hidden neurons.
/// Given a target mean activation \f$ \rho \f$ the mean activation of hidden neuron j over the whole dataset
/// \f$ \rho_j\f$ is interpreted as the activation propability and penalized using the KL-divergence:
/// \f$ KL(\rho||\rho_j) = \rho log(\frac{\rho}{\rho_j})+(1-\rho) log(\frac{1-\rho}{1-\rho_j}) \f$
///
/// This Error Function has two meta-parameters: rho governs the desired mean activation and
/// beta the strength of regularization. Another regularizer can be added using setRegularizer as in typical ErrorFunctions.
class SparseAutoencoderError : public SingleObjectiveFunction
{
public:
	typedef LabeledData<RealVector, RealVector> DatasetType;
	template<class HiddenNeuron, class OutputNeuron>
	SparseAutoencoderError(
		DatasetType const& dataset,
		Autoencoder<HiddenNeuron,OutputNeuron>* model, 
		AbstractLoss<RealVector, RealVector>* loss, 
		double rho = 0.5, double beta = 0.1
	);
	template<class HiddenNeuron, class OutputNeuron>
	SparseAutoencoderError(
		DatasetType const& dataset,
		TiedAutoencoder<HiddenNeuron,OutputNeuron>* model, 
		AbstractLoss<RealVector, RealVector>* loss, 
		double rho = 0.5, double beta = 0.1
	);

	SparseAutoencoderError& operator=(SparseAutoencoderError const& op){
		mp_wrapper.reset(op.mp_wrapper->clone());
		return *this;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SparseAutoencoderError"; }

	std::size_t numberOfVariables()const{
		return mp_wrapper->numberOfVariables();
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		mp_wrapper->proposeStartingPoint(startingPoint);
	}
	
	void setRegularizer(double factor, SingleObjectiveFunction* regularizer){
		m_regularizer = regularizer;
		m_regularizationStrength = factor;
	}

	double eval(RealVector const& input) const{
		m_evaluationCounter++;
		double value = mp_wrapper -> eval(input);
		if(m_regularizer)
			value += m_regularizationStrength * m_regularizer->eval(input);
		return value;
	}
	ResultType evalDerivative( SearchPointType const& input, FirstOrderDerivative & derivative ) const{
		m_evaluationCounter++;
		double value = mp_wrapper -> evalDerivative(input,derivative);
		if(m_regularizer){
			FirstOrderDerivative regularizerDerivative;
			value += m_regularizationStrength * m_regularizer->evalDerivative(input,regularizerDerivative);
			noalias(derivative) += m_regularizationStrength*regularizerDerivative;
		}
		return value;
	}

	friend void swap(SparseAutoencoderError& op1, SparseAutoencoderError& op2){
		swap(op1.mp_wrapper,op2.mp_wrapper);
	}

private:
	boost::scoped_ptr<detail::FunctionWrapperBase > mp_wrapper;

	SingleObjectiveFunction* m_regularizer;
	double m_regularizationStrength;
};

}
#include "Impl/SparseAutoencoderError.inl"
#endif
