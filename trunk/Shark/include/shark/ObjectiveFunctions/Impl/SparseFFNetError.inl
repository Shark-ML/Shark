/*!
 *  \brief implementation of basic error function
 *
 *  \author O.Krause
 *  \date 2012
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_SPARSEFFNETERROR_INL
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_SPARSEFFNETERROR_INL

namespace shark{
namespace detail{
///\brief Implementation of the SparseFFNetError
template<class HiddenNeuron,class OutputNeuron>
class SparseFFNetErrorWrapper:public FunctionWrapperBase{
private:
	typedef LabeledData<RealVector,RealVector>::const_batch_reference const_reference;
	///\brief calculates KL error
	double errorKL(RealVector const& meanActivation) const{
		std::size_t hiddens = meanActivation.size();
		double logRho = std::log(m_rho);
		double log1mRho = std::log(1.0-m_rho);

//		//sum of log(p/p_j)=-log(p_j)+log(p)
//		double t1 = -sum(log(meanActivation))+hiddens*logRho;
//		//sum of log((1-p)/(1-p_j))=-log(1-p_j)+log(1-p)
//		double t2 = -sum(log(RealScalarVector(hiddens,1.0)-meanActivation))+hiddens*log1mRho;
//		//return KL-error
//		return  m_rho*t1+(1.0-m_rho)*t2;

		double error = 0;
		for(std::size_t i = 0; i != hiddens; ++i){
			double a = meanActivation(i);

			if(a < 1.e-15){
				error += m_rho*(logRho-std::log(1.e-15));
			}else{
				error += m_rho*(logRho-std::log(a));
			}
			if((1-a) < 1.e-15){
				error += (1-m_rho)*(log1mRho-std::log(1.e-15));
			}else{
				error += (1-m_rho)*(log1mRho-std::log(1-a));
			}
		}
		return error;
	}

	///\brief Optimized implementation for FFNetworks with a single hidden layer.
	///
	///It uses the fact, that the KL-divergence gradient change is the same for every datapoint
	///so backpropagation can be delayed
	double evalDerivativeSingle(SearchPointType const& point, SearchPointType& gradient) const{
		typename Batch<RealVector>::type prediction;
		RealVector dataGradient(mep_model->numberOfParameters());
		ensure_size(gradient,mep_model->numberOfParameters());

		typename Batch<RealVector>::type errorDerivative;

		std::size_t inputs = mep_model->inputSize();
		std::size_t outputs = mep_model->outputSize();
		std::size_t neurons = mep_model->numberOfNeurons() - inputs;
		std::size_t hiddens = neurons - outputs;
		std::size_t dataSize = m_dataset.numberOfElements();

		RealVector meanActivation(hiddens);
		RealMatrix W1Derivatives(hiddens,inputs); // hiddenDerivatives * input
		RealVector hiddenDerivativeSum(hiddens);

		gradient.clear();
		meanActivation.clear();
		W1Derivatives.clear();
		hiddenDerivativeSum.clear();
		//create an object of the type of the hidden neurons
		HiddenNeuron hidden;

		boost::shared_ptr<State> state = mep_model->createState();
		double error = 0.0;
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			// calculate model output for the batch as well as the derivative
			mep_model->eval(batch.input, prediction,*state);

			// calculate error derivative of the loss function
			error += mep_loss->evalDerivative(batch.label, prediction,errorDerivative);

			//calculate derivative
			mep_model->weightedParameterDerivative(batch.input,errorDerivative,*state,dataGradient);
			noalias(gradient) += dataGradient;

			//now prepare KL-divergence step

			//get part of the responses which are the hidden neurons
			ConstRealSubMatrix hiddenActivation = rows(mep_model->neuronResponses(*state),inputs,inputs + hiddens);
 			// update mean activation of hidden neurons
			meanActivation += sum_columns(hiddenActivation);
			// calculate derivative of the hidden neurons
			RealMatrix hiddenDerivative = hidden.derivative(hiddenActivation);
			//update sum of derivatives
			noalias(hiddenDerivativeSum) += sum_columns(hiddenDerivative);

			// Calculate the derivative with respect to the lower weight matrix
			axpy_prod(hiddenDerivative,batch.input,W1Derivatives,false);
		}
		error /= dataSize;
		meanActivation /= dataSize;

		//calculate derivative of the KL-divergence and scale the W1Derivatives with it
		//the result is the correct update of the derivative
		for(std::size_t i = 0; i != hiddens; ++i){
			double derivativeKL = 0;
			if(meanActivation(i) < 1.e-15){
				derivativeKL = (1-m_rho)-m_rho/(1.e-15);
			}else if(1-meanActivation(i) < 1.e-15){
				derivativeKL = (1-m_rho)/(1.e-15)-m_rho;
			}else{
				derivativeKL = (1-m_rho)/(1-meanActivation(i))-m_rho/meanActivation(i);
			}
			row(W1Derivatives,i) *= m_beta*derivativeKL;
			hiddenDerivativeSum(i) *= m_beta*derivativeKL;
		}

		if(m_beta != 0){
			//now update the derivative of the first layer of the FFNet
			std::size_t W1params = hiddens*inputs;
			noalias(to_matrix(subrange(gradient,0,W1params),hiddens,inputs)) += W1Derivatives;

			//adjust bias units
			if(!mep_model->bias().empty()){
				std::size_t biasStart = inputs*hiddens+hiddens*outputs;
				noalias(subrange(gradient,biasStart,biasStart+hiddens)) += hiddenDerivativeSum;
			}
		}	
		gradient /= dataSize;

		// add kl error term to the error
		if(m_beta > 1.e-15)
			error += m_beta*errorKL(meanActivation);
		return error;
	}

	///\brief Implementation for FFNetworks with multiple hidden layers.
	double evalDerivativeMultiple(SearchPointType const& point, SearchPointType& gradient) const{
		typename Batch<RealVector>::type prediction;
		RealVector dataGradient(mep_model->numberOfParameters());
		typename Batch<RealVector>::type errorDerivative;

		std::size_t inputs = mep_model->inputSize();
		std::size_t outputs = mep_model->outputSize();
		std::size_t neurons = mep_model->numberOfNeurons();
		std::size_t hiddens = neurons - inputs - outputs;
		std::size_t dataSize = m_dataset.numberOfElements();

		RealVector meanActivation(hiddens);
		meanActivation.clear();

		boost::shared_ptr<State> state = mep_model->createState();
		//first calculate mean activation
		//we need the mean to calculate the derivative. so we have to evaluate everything twice.
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			// calculate model output for the batch as well as the derivative
			mep_model->eval(batch.input, prediction,*state);

			//get the submatrix of activations of the hidden neurons
			//and sum their activation to every pattern of the batch
			meanActivation+=sum_columns(
				rows(mep_model->neuronResponses(*state),inputs,inputs + hiddens)
			);
		}
		meanActivation /= dataSize;

		//calculate KL-derivative from the mean
		RealVector derivativeKL(hiddens);
		for(std::size_t i = 0; i != hiddens; ++i){
			derivativeKL(i) = (1-m_rho)/(1-meanActivation(i))-m_rho/meanActivation(i);
		}
		derivativeKL *= m_beta;

		double error=0.0;
		//now calculate the derivative and the erorr in a second pass
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			// calculate model output for the batch as well as the derivative
			mep_model->eval(batch.input, prediction,*state);

			// calculate error derivative of the loss function
			error += mep_loss->evalDerivative(batch.label, prediction,errorDerivative);

			//initialize delta
			RealMatrix delta(neurons,boost::size(batch));
			//set coefficients
			noalias(rows(delta,inputs+hiddens,neurons)) = trans(errorDerivative);
			//set KL-penalty of hidden neurons
			noalias(rows(delta,inputs,inputs+hiddens)) = trans(repeat(derivativeKL,boost::size(batch)));

			//calculate derivative
			mep_model->weightedParameterDerivativeFullDelta(batch.input,delta,*state,dataGradient);
			gradient += dataGradient;
		}
		error /= dataSize;
		if(m_beta > 1.e-15)
			error += m_beta * errorKL(meanActivation);
		gradient /= dataSize;
		return error;
	}
	
public:
	typedef FFNet<HiddenNeuron, OutputNeuron> Network;
	SparseFFNetErrorWrapper(
		LabeledData<RealVector, RealVector> const& dataset,
		Network* model, AbstractLoss<RealVector, RealVector>* loss,
		double rho, double beta
	):m_dataset(dataset), m_rho(rho), m_beta(beta) {
		SHARK_ASSERT(model!=NULL);
		SHARK_ASSERT(loss!=NULL);
		mep_model = model;
		mep_loss = loss;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "SparseFFNetErrorWrapper"; }

	FunctionWrapperBase* clone()const{
		return new SparseFFNetErrorWrapper<HiddenNeuron, OutputNeuron>(*this);
	}

	void configure( const PropertyTree & node ) {
		PropertyTree::const_assoc_iterator it = node.find("model");
		if(it!=node.not_found())
		{
			mep_model->configure(it->second);
		}
		m_rho = node.get<double>("rho",m_rho);
		m_beta = node.get<double>("beta",m_beta);
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint = mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_model->numberOfParameters();
	}

	double eval(RealVector const& input) const {
		size_t dataSize = m_dataset.numberOfElements();
		std::size_t inputs = mep_model->inputSize();
		std::size_t neurons = mep_model->numberOfNeurons() - inputs;
		std::size_t hiddens = neurons - mep_model->outputSize();

		mep_model->setParameterVector(input);

		RealVector meanActivation(hiddens);
		meanActivation.clear();
		boost::shared_ptr<State> state = mep_model->createState();

		typename Batch<RealVector>::type prediction;
		double error = 0.0;
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			mep_model->eval(batch.input, prediction,*state);
			error += mep_loss->eval(batch.label, prediction);
			//get the submatrix of activations of the hidden neurons and sum their activation to every pattern of the batch
			meanActivation+=sum_columns(
				rows(mep_model->neuronResponses(*state),inputs,inputs+hiddens)
			);
		}
		meanActivation /= dataSize;
		error /= dataSize;
		if(m_beta > 1.e-15)
			error += m_beta * errorKL(meanActivation);

		return error;
	}

	ResultType evalDerivative( SearchPointType const& point, FirstOrderDerivative & derivative ) const {
		mep_model->setParameterVector(point);

		derivative.resize(mep_model->numberOfParameters());
		derivative.clear();

		//check the number of hidden layers
		if(mep_model->layerMatrices().size() == 2)
			return evalDerivativeSingle(point, derivative);
		else
			return evalDerivativeMultiple(point, derivative);
	}

private:
	Network* mep_model;
	AbstractLoss<RealVector, RealVector>* mep_loss;
	LabeledData<RealVector,RealVector> m_dataset;
	double m_rho;
	double m_beta;
};


} // namespace detail

template<class HiddenNeuron,class OutputNeuron>
SparseFFNetError::SparseFFNetError(
	DatasetType const& dataset,
	FFNet<HiddenNeuron, OutputNeuron>* model, 
	AbstractLoss<RealVector, RealVector>* loss, double rho, double beta
){
	m_features |= HAS_FIRST_DERIVATIVE;
	m_features |= CAN_PROPOSE_STARTING_POINT;

	mp_wrapper.reset(new detail::SparseFFNetErrorWrapper<HiddenNeuron, OutputNeuron>(dataset,model,loss,rho,beta));
}
}
#endif
