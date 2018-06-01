//===========================================================================
/*!
 *
 *
 * \brief       Generic Stochastic Average Gradient Descent training for linear models
 *
 *
 *
 *
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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


#ifndef SHARK_ALGORITHMS_LinearSAGTrainer_H
#define SHARK_ALGORITHMS_LinearSAGTrainer_H


#include <shark/Algorithms/Trainers/AbstractWeightedTrainer.h>
#include <shark/Core/IParameterizable.h>
#include <shark/Models/LinearModel.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/Statistics/Distributions/MultiNomialDistribution.h>
#include <shark/Data/DataView.h>


namespace shark
{
	
namespace detail{
	template<class InputType, class LabelType>
	struct LinearSAGTrainerBase{
		typedef AbstractWeightedTrainer< LinearModel<InputType>,LabelType > type;
		typedef AbstractLoss<LabelType,RealVector> LossType;
	};
	template<class InputType>
	struct LinearSAGTrainerBase<InputType, unsigned int>{
		typedef AbstractWeightedTrainer< LinearClassifier<InputType>, unsigned int > type;
		typedef AbstractLoss<unsigned int,RealVector> LossType;
	};
}


///
/// \brief Stochastic Average Gradient Method for training of linear models,
///
/// Given a differentiable loss function L(f, y) and a model f_j(x)= w_j^Tx+b 
/// this trainer solves the regularized risk minimization problem
/// \f[
///     \min \frac{1}{2} \sum_j \frac{\lambda}{2}\|w_j\|^2 + \frac 1 {\ell} \sum_i L(y_i, f(x_i)),
/// \f]
/// where i runs over training data, j over the model outputs, and lambda > 0 is the
/// regularization parameter. 
///
/// The algorithm uses averaging of the algorithm to obtain a good estimate of the gradient.
/// Averaging is performed by summing over the last gradient value obtained for each data point.
/// At the beginning this estimate is far off as old gradient values are outdated, but as the
/// algorithm converges, this gives linear convergence on strictly convex functions
/// and O(1/T) convergence on not-strictly convex functions.
///
/// The algorithm supports classification and regresseion, dense and sparse inputs
/// and weighted and unweighted datasets
/// Reference:
/// Schmidt, Mark, Nicolas Le Roux, and Francis Bach.
/// "Minimizing finite sums with the stochastic average gradient."
/// arXiv preprint arXiv:1309.2388 (2013).
/// \ingroup supervised_trainer
template <class InputType, class LabelType>
class LinearSAGTrainer : public detail::LinearSAGTrainerBase<InputType,LabelType>::type, public IParameterizable<>
{
private:
	typedef typename detail::LinearSAGTrainerBase<InputType,LabelType>::type Base;
public:
	typedef typename Base::ModelType ModelType;
	typedef typename Base::WeightedDatasetType WeightedDatasetType;
	typedef typename detail::LinearSAGTrainerBase<InputType,LabelType>::LossType LossType;


	/// \brief Constructor
	///
	/// \param  loss            (sub-)differentiable loss function
	/// \param  lambda          regularization parameter fort wo-norm regularization, 0 by default
	/// \param  offset          whether to train with offset/bias parameter or not, default is true
	LinearSAGTrainer(LossType const* loss, double lambda = 0, bool offset = true)
	: mep_loss(loss)
	, m_lambda(lambda)
	, m_offset(offset)
	, m_maxEpochs(0)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LinearSAGTrainer"; }

	using Base::train;
	void train(ModelType& model, WeightedDatasetType const& dataset){
		trainImpl(random::globalRng, model,dataset,*mep_loss);
	}
	

	/// \briefReturn the number of training epochs.
	/// A value of 0 indicates that the default of max(10, dimensionOfData) should be used.
	std::size_t epochs() const
	{ return m_maxEpochs; }

	/// \brief Set the number of training epochs.
	/// A value of 0 indicates that the default of max(10, dimensionOfData) should be used.
	void setEpochs(std::size_t value)
	{ m_maxEpochs = value; }


	/// \brief Return the value of the regularization parameter lambda.
	double lambda() const
	{ return m_lambda; }
	
	/// \brief Set the value of the regularization parameter lambda.
	void setLambda(double lambda)
	{ m_lambda = lambda; }

	/// \brief Check whether the model to be trained should include an offset term.
	bool trainOffset() const
	{ return m_offset; }
	
	/// \brief Sets whether the model to be trained should include an offset term.
	void setTrainOffset(bool offset)
	{ m_offset = offset;}

	/// \brief Returns the vector of hyper-parameters(same as lambda)
	RealVector parameterVector() const
	{
		return RealVector(1,m_lambda);
	}

	/// \brief Sets the vector of hyper-parameters(same as lambda)
	void setParameterVector(RealVector const& newParameters)
	{
		SIZE_CHECK(newParameters.size() == 1);
		m_lambda = newParameters(0);
	}

	///\brief Returns the number of hyper-parameters.
	size_t numberOfParameters() const
	{
		return 1;
	}

private:
	//initializes the model in the classification case and calls iterate to train it
	void trainImpl(
		random::rng_type& rng,
		LinearClassifier<InputType>& classifier,
		WeightedLabeledData<InputType, unsigned int> const& dataset,
		AbstractLoss<unsigned int,RealVector> const& loss
	){
		//initialize model
		std::size_t classes = numberOfClasses(dataset);
		if(classes == 2) classes = 1;//special case: 2D classification is always encoded by the sign of the output
		std::size_t dim = inputDimension(dataset);
		auto& model = classifier.decisionFunction();
		model.setStructure(dim,classes, m_offset);
		
		iterate(rng, model,dataset,loss);
	}
	//initializes the model in the regression case and calls iterate to train it
	template<class LabelT>
	void trainImpl(
		random::rng_type& rng,
		LinearModel<InputType>& model,
		WeightedLabeledData<InputType, LabelT> const& dataset,
		AbstractLoss<LabelT,RealVector> const& loss
	){
		//initialize model
		std::size_t labelDim = labelDimension(dataset);
		std::size_t dim = inputDimension(dataset);
		model.setStructure(dim,labelDim, m_offset);
		iterate(rng, model,dataset,loss);
	}
	
	//dense vector case is easier, mostly implemented for simple exposition of the algorithm
	template<class T>
	void iterate(
		random::rng_type& rng,
		LinearModel<blas::vector<T> >& model,
		WeightedLabeledData<blas::vector<T>, LabelType> const& dataset,
		AbstractLoss<LabelType,RealVector> const& loss
	){
		
		//get stats of the dataset
		DataView<LabeledData<InputType, LabelType> const> data(dataset.data());
		std::size_t ell = data.size();
		std::size_t labelDim = model.outputShape().numElements();
		std::size_t dim = model.inputShape().numElements();
		
		//set number of iterations
		std::size_t iterations = m_maxEpochs * ell;
		if(m_maxEpochs == 0)
			iterations = std::max(10 * ell, std::size_t(std::ceil(dim * ell)));
		
		//picking distribution picks proportional to weight
		RealVector probabilities = createBatch(dataset.weights().elements());
		probabilities /= sum(probabilities);
		MultiNomialDistribution dist(probabilities);
			
		//variables used for the SAG loop
		RealMatrix gradD(labelDim,ell,0); // gradients of regularized loss minimization with a linear model have the form sum_i D_i*x_i. We store the last acquired estimate
		RealMatrix grad(labelDim,dim);// gradient of the weight matrix.
		RealVector gradOffset(labelDim,0); //sum_i D_i, gradient estimate for the offset
		RealVector pointNorms(ell); //norm of each point in the dataset
		for(std::size_t  i = 0; i != ell; ++i){
			pointNorms(i) = norm_sqr(data[i].input);
		}
		// preinitialize everything to prevent costly memory allocations in the loop
		RealVector f_b(labelDim, 0.0); // prediction of the model
		RealVector derivative(labelDim, 0.0); //derivative of the loss
		double L = 1; // initial estimate for the lipschitz-constant
		
		// SAG loop
		for(std::size_t iter = 0; iter < iterations; iter++)
		{
			// choose data point
			std::size_t b = dist(rng);
			
			// compute prediction
			noalias(f_b) = prod(model.matrix(), data[b].input);
			if(m_offset) noalias(f_b) += model.offset();
			
			// compute loss gradient
			double currentValue = loss.evalDerivative(data[b].label, f_b, derivative);
			
			//update gradient (needs to be multiplied with kappa)
			noalias(grad) += probabilities(b) * outer_prod(derivative-column(gradD,b), data[b].input);
			if(m_offset) noalias(gradOffset) += probabilities(b) *(derivative-column(gradD,b));
			noalias(column(gradD,b)) = derivative; //we got a new estimate for D of element b.
			
			// update gradient
			double eta = 1.0/(L+m_lambda);
			noalias(model.matrix()) *= 1 - eta * m_lambda;//2-norm regularization
			for(std::size_t i = 0; i != labelDim; ++i){
				for(std::size_t j = 0; j != dim; ++j){
					model.matrix()(i,j) -= eta*grad(i,j);
				}
			}
			//~ noalias(model.matrix()) -= eta * grad;
			if(m_offset) noalias(model.offset()) -= eta * gradOffset;
			
			//line-search procedure, 4.6 in the paper
			noalias(f_b) -= derivative/L*pointNorms(b);
			double newValue = loss.eval(data[b].label, f_b);
			if(norm_sqr(derivative)*pointNorms(b) > 1.e-8 && newValue > currentValue - 1/(2*L)*norm_sqr(derivative)*pointNorms(b)){
				L *= 2;
			}
			L*= std::pow(2.0,-1.0/ell);//allow L to slightly shrink in case our initial estimate was too large
		}
	}
	
	template<class T>
	void iterate(
		random::rng_type& rng,
		LinearModel<blas::compressed_vector<T> >& model,
		WeightedLabeledData<blas::compressed_vector<T>, LabelType> const& dataset,
		AbstractLoss<LabelType,RealVector> const& loss
	){
		
		//get stats of the dataset
		DataView<LabeledData<InputType, LabelType> const> data(dataset.data());
		std::size_t ell = data.size();
		std::size_t labelDim = model.outputSize();
		std::size_t dim = model.inputSize();
		
		//set number of iterations
		std::size_t iterations = m_maxEpochs * ell;
		if(m_maxEpochs == 0)
			iterations = std::max(10 * ell, std::size_t(std::ceil(dim * ell)));
		
		//picking distribution picks proportional to weight
		RealVector probabilities = createBatch(dataset.weights().elements());
		probabilities /= sum(probabilities);
		MultiNomialDistribution dist(probabilities);
			
		//variables used for the SAG loop
		blas::matrix<double,blas::column_major> gradD(labelDim,ell,0); // gradients of regularized loss minimization with a linear model have the form sum_i D_i*x_i. We store the last acquired estimate
		RealMatrix grad(labelDim,dim);// gradient of the weight matrix.
		RealVector gradOffset(labelDim,0); //sum_i D_i, gradient estimate for the offset
		RealVector pointNorms(ell); //norm of each point in the dataset
		for(std::size_t  i = 0; i != ell; ++i){
			pointNorms(i) = norm_sqr(data[i].input);
		}
		// preinitialize everything to prevent costly memory allocations in the loop
		RealVector f_b(labelDim, 0.0); // prediction of the model
		RealVector derivative(labelDim, 0.0); //derivative of the loss
		double kappa =1; //we store the matrix as kappa*model.matrix() where kappa stores the effect of the 2-norm regularisation
		double L = 1; // initial estimate for the lipschitz-constant
		
		//just in time updating for sparse inputs.
		//we need to store a running sum of step lengths which we can then apply whenever an index got updated
		RealVector appliedRates(dim,0.0);//applied steps since the last reset for each dimension
		double stepsCumSum = 0.0;// cumulative update steps
		
		
		// SAG loop
		for(std::size_t iter = 0; iter < iterations; iter++)
		{
			// choose data point
			std::size_t b = dist(rng);
			auto const& point = data[b];
			
			//just in time update of the previous steps for every nonzero element of point b
			auto end = point.input.end();
			for(auto  pos = point.input.begin(); pos != end; ++pos){
				std::size_t index = pos.index();
				noalias(column(model.matrix(),index)) -= (stepsCumSum - blas::repeat(appliedRates(index),labelDim))*column(grad,index);
				appliedRates(index) = stepsCumSum;
			}
			
			// compute prediction
			noalias(f_b) = kappa * prod(model.matrix(), point.input);
			if(m_offset) noalias(f_b) += model.offset();
			
			// compute loss gradient
			double currentValue = loss.evalDerivative(point.label, f_b, derivative);
			
			//update gradient (needs to be multiplied with kappa)
			//~ noalias(grad) += probabilities(b) * outer_prod(derivative-column(gradD,b), point.input); //todo: this is slow for some reason.
			for(std::size_t l = 0; l != derivative.size(); ++l){
				double val = probabilities(b) * (derivative(l) - gradD(l,b));
				noalias(row(grad,l)) += val * point.input;
			}
			
			if(m_offset) noalias(gradOffset) += probabilities(b) *(derivative-column(gradD,b));
			noalias(column(gradD,b)) = derivative; //we got a new estimate for D of element b.
			
			// update gradient
			double eta = 1.0/(L+m_lambda);
			stepsCumSum += kappa * eta;//we delay update of the matrix
			if(m_offset) noalias(model.offset()) -= eta * gradOffset;
			kappa *= 1 - eta * m_lambda;//2-norm regularization
			
			//line-search procedure, 4.6 in the paper
			noalias(f_b) -= derivative/L*pointNorms(b);
			double newValue = loss.eval(point.label, f_b);
			if(norm_sqr(derivative)*pointNorms(b) > 1.e-8 && newValue > currentValue - 1/(2*L)*norm_sqr(derivative)*pointNorms(b)){
				L *= 2;
			}
			L*= std::pow(2.0,-1.0/ell);//allow L to slightly shrink in case our initial estimate was too large
			
			
			//every epoch we reset the internal variables for numeric stability and apply all outstanding updates
			//this also ensures that in the last iteration all updates are applied (as iterations is a multiple of ell)
			if((iter +1)% ell == 0){
				noalias(model.matrix()) -= (stepsCumSum - blas::repeat(appliedRates,labelDim))*grad;
				model.matrix() *= kappa;
				kappa = 1;
				stepsCumSum = 0.0;
				appliedRates.clear();
			}
		}
	}
	
	LossType const* mep_loss;                 ///< pointer to loss function
	double m_lambda;                          ///< regularization parameter
	bool m_offset;                            ///< should the resulting model have an offset term?
	std::size_t m_maxEpochs;                  ///< number of training epochs (sweeps over the data), or 0 for default = max(10, C)
};

}
#endif
