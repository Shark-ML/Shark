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
#ifndef SHARK_OBJECTIVEFUNCTIONS_IMPL_SPARSE_AUTOENCODER_ERROR_INL
#define SHARK_OBJECTIVEFUNCTIONS_IMPL_SPARSE_AUTOENCODER_ERROR_INL

namespace shark{
namespace detail{
///\brief Implementation of the SparseAutoencoderError
template<class Network>
class SparseAutoencoderErrorWrapper:public FunctionWrapperBase{
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
	
public:
	SparseAutoencoderErrorWrapper(
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
	{ return "SparseAutoencoderErrorWrapper"; }

	FunctionWrapperBase* clone()const{
		return new SparseAutoencoderErrorWrapper<Network>(*this);
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
		std::size_t hiddens = mep_model->numberOfHiddenNeurons();

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
			meanActivation+=sum_rows(mep_model->hiddenResponses(*state));
		}
		meanActivation /= dataSize;
		error /= dataSize;
		if(m_beta > 1.e-15)
			error += m_beta * errorKL(meanActivation);

		return error;
	}

	ResultType evalDerivative( SearchPointType const& point, FirstOrderDerivative & gradient ) const {
		mep_model->setParameterVector(point);

		gradient.resize(mep_model->numberOfParameters());
		gradient.clear();

		typename Batch<RealVector>::type prediction;
		RealVector dataGradient(mep_model->numberOfParameters());
		ensure_size(gradient,mep_model->numberOfParameters());

		typename Batch<RealVector>::type errorDerivative;

		std::size_t inputs = mep_model->inputSize();
		std::size_t hiddens = mep_model->numberOfHiddenNeurons();
		std::size_t dataSize = m_dataset.numberOfElements();

		RealVector meanActivation(hiddens,0.0);
		RealMatrix W1Derivatives(hiddens,inputs,0.0); // hiddenDerivatives * input
		RealVector hiddenDerivativeSum(hiddens,0.0);
		gradient.clear();
		
		boost::shared_ptr<State> state = mep_model->createState();
		double error = 0.0;
		BOOST_FOREACH(const_reference batch,m_dataset.batches()){
			// calculate model output for the batch as well as the gradient
			mep_model->eval(batch.input, prediction,*state);

			// calculate error gradient of the loss function
			error += mep_loss->evalDerivative(batch.label, prediction,errorDerivative);

			//calculate gradient
			mep_model->weightedParameterDerivative(batch.input,errorDerivative,*state,dataGradient);
			noalias(gradient) += dataGradient;

			//now prepare KL-divergence step

			//get part of the responses which are the hidden neurons
			RealMatrix const& hiddenActivation = mep_model->hiddenResponses(*state);
 			// update mean activation of hidden neurons
			meanActivation += sum_rows(hiddenActivation);
			// calculate gradient of the hidden neurons
			RealMatrix hiddenDerivative = mep_model->hiddenActivationFunction().derivative(hiddenActivation);
			//update sum of derivatives
			noalias(hiddenDerivativeSum) += sum_rows(hiddenDerivative);

			// Calculate the gradient with respect to the lower weight matrix
			axpy_prod(trans(hiddenDerivative),batch.input,W1Derivatives,false);
		}
		error /= dataSize;
		meanActivation /= dataSize;

		//calculate gradient of the KL-divergence and scale the W1Derivatives with it
		//the result is the correct update of the gradient
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
			//now update the gradient of the first layer of the FFNet
			std::size_t W1params = hiddens*inputs;
			noalias(to_matrix(subrange(gradient,0,W1params),hiddens,inputs)) += W1Derivatives;
			
			//adjust bias units
			std::size_t biasStart = mep_model->numberOfParameters()-inputs-hiddens;
			noalias(subrange(gradient,biasStart,biasStart+hiddens)) += hiddenDerivativeSum;
		
		}	
		
		gradient /= dataSize;

		// add kl error term to the error
		if(m_beta > 1.e-15)
			error += m_beta*errorKL(meanActivation);
		return error;
	}

private:
	Network* mep_model;
	AbstractLoss<RealVector, RealVector>* mep_loss;
	LabeledData<RealVector,RealVector> m_dataset;
	double m_rho;
	double m_beta;
};


} // namespace detail

template<class HiddenNeuron, class OutputNeuron>
SparseAutoencoderError::SparseAutoencoderError(
	DatasetType const& dataset,
	Autoencoder<HiddenNeuron,OutputNeuron>* model, 
	AbstractLoss<RealVector, RealVector>* loss, 
	double rho, double beta
):m_regularizer(0){
	m_features |= HAS_FIRST_DERIVATIVE;
	m_features |= CAN_PROPOSE_STARTING_POINT;

	mp_wrapper.reset(new detail::SparseAutoencoderErrorWrapper<Autoencoder<HiddenNeuron,OutputNeuron> >(dataset,model,loss,rho,beta));
}

template<class HiddenNeuron, class OutputNeuron>
SparseAutoencoderError::SparseAutoencoderError(
	DatasetType const& dataset,
	TiedAutoencoder<HiddenNeuron,OutputNeuron>* model, 
	AbstractLoss<RealVector, RealVector>* loss, 
	double rho, double beta
):m_regularizer(0){
	m_features |= HAS_FIRST_DERIVATIVE;
	m_features |= CAN_PROPOSE_STARTING_POINT;

	mp_wrapper.reset(new detail::SparseAutoencoderErrorWrapper<TiedAutoencoder<HiddenNeuron,OutputNeuron> >(dataset,model,loss,rho,beta));
}
}
#endif
