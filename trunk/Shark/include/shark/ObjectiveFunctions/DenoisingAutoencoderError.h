/*!
 *  \brief Error Function used for training Denoising Autoencoders
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_DENOISINGAUTOENCODERERROR_H
#define SHARK_OBJECTIVEFUNCTIONS_DENOISINGAUTOENCODERERROR_H

#include <shark/Models/AbstractModel.h>
#include <shark/ObjectiveFunctions/Loss/AbstractLoss.h>
#include <shark/ObjectiveFunctions/DataObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

#include <boost/range/algorithm_ext/iota.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>
namespace shark{

///
/// \brief Objective function for unsupervised training of denoising autoencoders.
///
/// \par
/// An Autoencoder is a model which is trained to reconstruct it's input. The idea is
/// to decompose the model after training into an encoding and decoding structure 
/// using the encoder to project the input data into a different space. A loss function
/// is used to measure the distance of the reconstruction of the input with the input.
/// Common choices are the cross entropy loss or squared loss.
/// 
/// \par 
/// The main difference of an denoising autoencoder from a normal autoencoder is, that the 
/// input during training is noisy. This prevents typical problems which stem from overcomplete
/// representations of the encoding. The noise used is setting k random inputs of every input
/// vector to 0. If set to 0, no noise is added which is usefull if no overcomplete representation
/// is trained.
template<class InputType = RealVector,class RngType = Rng::rng_type>
class DenoisingAutoencoderError : public UnsupervisedObjectiveFunction<InputType>
{
public:
	typedef UnsupervisedObjectiveFunction<InputType> base_type;
	typedef typename base_type::SearchPointType SearchPointType;
	typedef typename base_type::ResultType ResultType;
	typedef typename base_type::FirstOrderDerivative FirstOrderDerivative;
	typedef typename base_type::SecondOrderDerivative SecondOrderDerivative;

	DenoisingAutoencoderError(
		AbstractModel<InputType,InputType>* model, 
		AbstractLoss<InputType, InputType>* loss, 
		std::size_t k = 0,
		RngType& rng = BaseRng<RngType>::globalRng
	):mep_model(model),mep_loss(loss),m_k(k),m_uni(rng,0,1){
		if(mep_model->hasFirstParameterDerivative() && mep_loss->hasFirstDerivative())
			this->m_features|=base_type::HAS_FIRST_DERIVATIVE;
		this->m_features|=base_type::CAN_PROPOSE_STARTING_POINT;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "DenoisingAutoencoderError"; }

	void configure(const PropertyTree & node){
		m_k=node.get<std::size_t>("k",0);
		PropertyTree::const_assoc_iterator it = node.find("model");
		if(it!=node.not_found())
		{
			mep_model->configure(it->second);
		}
		it = node.find("loss");
		if(it!=node.not_found())
		{
			mep_loss->configure(it->second);
		}
	}
	void setData(const UnlabeledData<InputType>& data){
		m_data = data;
		m_dataDimension = dataDimension(m_data);
	}
	
	void setK(std::size_t newK){
		m_k = newK;
	}

	void proposeStartingPoint(SearchPointType& startingPoint) const{
		startingPoint=mep_model->parameterVector();
	}
	
	std::size_t numberOfVariables()const{
		return mep_model->numberOfParameters();
	}

	double eval(RealVector const& input) const{
		this->m_evaluationCounter++;
		mep_model->setParameterVector(input);
		
		typedef typename AbstractModel<InputType, InputType>::BatchInputType BatchType;
		//create vector of indizes 0...m_dataDimension-1
		std::vector<std::size_t> indizes(m_dataDimension);
		boost::range::iota(indizes,0);
		
		double error = 0;
		BatchType predictions;
		for(std::size_t i = 0; i != m_data.numberOfBatches(); ++i){
			if(m_k == 0){
				mep_model->eval(m_data.batch(i),predictions);
				error+=mep_loss->eval(m_data.batch(i), predictions);
			}else{
				BatchType noisyBatch = m_data.batch(i);
				//null elements in the batch
				for(std::size_t j = 0; j != size(noisyBatch); ++j){
					boost::range::random_shuffle(indizes,m_uni);
					for(std::size_t elem = 0; elem != m_k; ++elem){
						get(noisyBatch,j)[indizes[elem]]=0;
					}
				}
				mep_model->eval(noisyBatch,predictions);
				error+=mep_loss->eval(m_data.batch(i), predictions);
			}
		}
		error/=m_data.numberOfElements();
		return error;
	}
	ResultType evalDerivative( 
		const SearchPointType & input, 
		FirstOrderDerivative & derivative 
	) const{
		this->m_evaluationCounter++;

		mep_model->setParameterVector(input);
		boost::shared_ptr<State> state = mep_model->createState();
		
		ensure_size(derivative,mep_model->numberOfParameters());
		derivative.clear();
		
		typedef typename AbstractModel<InputType, InputType>::BatchInputType BatchType;
		
		//create vector of indizes 0...m_dataDimension-1
		std::vector<std::size_t> indizes(m_dataDimension);
		boost::range::iota(indizes,0);
		
		double error = 0;
		BatchType predictions;
		BatchType errorDerivative;
		RealVector dataGradient;
		for(std::size_t i = 0; i != m_data.numberOfBatches(); ++i){
			if(m_k == 0){
				mep_model->eval(m_data.batch(i),predictions,*state);

				// calculate error derivative of the loss function
				error += mep_loss->evalDerivative(m_data.batch(i), predictions,errorDerivative);

				//calculate the gradient using the chain rule
				mep_model->weightedParameterDerivative(m_data.batch(i),errorDerivative,*state,dataGradient);
				derivative+=dataGradient;
			}else{
				BatchType noisyBatch = m_data.batch(i);
				
				//null elements in the batch
				for(std::size_t j = 0; j != size(noisyBatch); ++j){
					boost::range::random_shuffle(indizes,m_uni);
					for(std::size_t elem = 0; elem != m_k; ++elem){
						get(noisyBatch,j)[indizes[elem]]=0;
					}
				}
				
				mep_model->eval(noisyBatch,predictions,*state);

				// calculate error derivative of the loss function
				error += mep_loss->evalDerivative(m_data.batch(i), predictions,errorDerivative);

				//calculate the gradient using the chain rule
				mep_model->weightedParameterDerivative(noisyBatch,errorDerivative,*state,dataGradient);
				derivative+=dataGradient;
			}
		}
		std::size_t dataSize=m_data.numberOfElements();
		error/=dataSize;
		derivative/=dataSize;
		return error;
	}

private:
	AbstractModel<InputType,InputType>* mep_model;
	AbstractLoss<InputType, InputType>* mep_loss;
	UnlabeledData<InputType> m_data;
	std::size_t m_k;
	std::size_t m_dataDimension;
	mutable DiscreteUniform<RngType> m_uni;
};

}
#endif
