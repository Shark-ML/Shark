/*
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
*/
#ifndef SHARK_UNSUPERVISED_RBM_RBM_H
#define SHARK_UNSUPERVISED_RBM_RBM_H

#include <shark/Models/AbstractModel.h>
#include <shark/Unsupervised/RBM/Energy.h>

#include <sstream>
#include <boost/serialization/string.hpp>
namespace shark{

///\brief stub for the RBM class. at the moment it is just a holder of the parameter set and the Energy.
template<class VisibleLayerT,class HiddenLayerT, class RngT>
class RBM : public AbstractModel<RealVector, RealVector>{
private:
	typedef AbstractModel<RealVector, RealVector> base_type;
public:
	typedef HiddenLayerT HiddenType; //< type of the hidden layer
	typedef VisibleLayerT VisibleType; //< type of the visible layer
	typedef RngT RngType;
	typedef Energy<RBM<VisibleType,HiddenType,RngT> > EnergyType;//< Type of the energy function
	
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	
private:
	/// \brief The weight matrix connecting hidden and visible layer.
	RealMatrix m_weightMatrix;

	///The layer of hidden Neurons
	HiddenType m_hiddenNeurons;

	///The Layer of visible Neurons
	VisibleType m_visibleNeurons;

	RngType* mpe_rng;
	bool m_forward;
	bool m_evalMean;

	///\brief Evaluates the input by propagating the visible input to the hidden neurons.
	///
	///@param patterns batch of states of visible units
	///@param outputs batch of (expected) states of hidden units
	void evalForward(BatchInputType const& state,BatchOutputType& output)const{
		std::size_t batchSize=state.size1();
		typename HiddenType::StatisticsBatch statisticsBatch(batchSize,numberOfHN());
		RealMatrix inputBatch(batchSize,numberOfHN());
		output.resize(state.size1(),numberOfHN());
		
		energy().inputHidden(inputBatch,state);
		hiddenNeurons().sufficientStatistics(inputBatch,statisticsBatch,blas::repeat(1.0,batchSize));

		if(m_evalMean){
			noalias(output) = hiddenNeurons().mean(statisticsBatch);
		}
		else{
			hiddenNeurons().sample(statisticsBatch,output,*mpe_rng);
		}
	}

	///\brief Evaluates the input by propagating the hidden input to the visible neurons.
	///
	///@param patterns batch of states of hidden units
	///@param outputs batch of (expected) states of visible units
	void evalBackward(BatchInputType const& state,BatchOutputType& output)const{
		std::size_t batchSize = state.size1();
		typename VisibleType::StatisticsBatch statisticsBatch(batchSize,numberOfVN());
		RealMatrix inputBatch(batchSize,numberOfVN());
		output.resize(batchSize,numberOfVN());
		
		energy().inputVisible(inputBatch,state);
		visibleNeurons().sufficientStatistics(inputBatch,statisticsBatch,blas::repeat(1.0,batchSize));
		
		if(m_evalMean){
			noalias(output) = visibleNeurons().mean(statisticsBatch);
		}
		else{
			visibleNeurons().sample(statisticsBatch,output,*mpe_rng);
		}
	}
public:
	RBM(RngType& rng):mpe_rng(&rng),m_forward(true),m_evalMean(true)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RBM"; }

	///\brief Returns the total number of parameters of the model.
	std::size_t numberOfParameters()const {
		std::size_t parameters = numberOfVN()*numberOfHN();
		parameters += m_hiddenNeurons.numberOfParameters();
		parameters += m_visibleNeurons.numberOfParameters();
		return parameters;
	}
	
	///\brief Returns the parameters of the Model as parameter vector.
	RealVector parameterVector () const {
		RealVector ret(numberOfParameters());
		init(ret) << toVector(m_weightMatrix),blas::parameters(m_hiddenNeurons),blas::parameters(m_visibleNeurons);
		return ret;
	};

	///\brief Sets the parameters of the model.
	///
	/// @param newParameters vector of parameters  
 	void setParameterVector(const RealVector& newParameters) {
		init(newParameters) >> toVector(m_weightMatrix),blas::parameters(m_hiddenNeurons),blas::parameters(m_visibleNeurons);
 	}
	
	///\brief Configures the structure.
	void configure( const PropertyTree & node ){
		size_t numberOfHN = node.get<unsigned int>("numberOfHN");
		size_t numberOfVN = node.get<unsigned int>("numberOfVN");
		setStructure(numberOfVN,numberOfHN);
	}
	
	///\brief Creates the structure of the RBM.
	///
	///@param hiddenNeurons number of hidden neurons.
	///@param visibleNeurons number of visible neurons.
	void setStructure(std::size_t visibleNeurons,std::size_t hiddenNeurons){
		m_weightMatrix.resize(hiddenNeurons,visibleNeurons);
		m_weightMatrix.clear();
		
		m_hiddenNeurons.resize(hiddenNeurons);
		m_visibleNeurons.resize(visibleNeurons);
	}
	
	///\brief Returns the layer of hidden neurons.
	HiddenType const& hiddenNeurons()const{
		return m_hiddenNeurons;
	}
	///\brief Returns the layer of hidden neurons.
	HiddenType& hiddenNeurons(){
		return m_hiddenNeurons;
	}
	///\brief Returns the layer of visible neurons.
	VisibleType& visibleNeurons(){
		return m_visibleNeurons;
	}
	///\brief Returns the layer of visible neurons.
	VisibleType const& visibleNeurons()const{
		return m_visibleNeurons;
	}
	
	///\brief Returns the weight matrix connecting the layers.
	RealMatrix& weightMatrix(){
		return m_weightMatrix;
	}
	///\brief Returns the weight matrix connecting the layers.
	RealMatrix const& weightMatrix()const{
		return m_weightMatrix;
	}
	
	///\brief Returns the energy function of the RBM.
	EnergyType energy()const{
		return EnergyType(m_visibleNeurons,m_hiddenNeurons,m_weightMatrix);
	}
	
	///\brief Returns the random number generator associated with this RBM.
	RngType& rng(){
		return *mpe_rng;
	}
	
	///\brief Sets the type of evaluation, eval will perform.
	///
	///Eval performs its operation based on the state of this function.
	///There are two ways to pass data through an rbm: either forward, setting the states of the
	///visible neurons and sample the hidden states or backwards, where the state of the hidden is fixed and the visible
	///are sampled. 
	///Instead of the state of the hidden/visible, one often wants the mean of the state \f$ E_{p(h|v)}\left(h\right)\f$. 
	///By default, the RBM uses the forward evaluation and returns the mean of the state
	///
	///@param forward whether the forward view should be used false=backwards
	///@param evalMean whether the mean state should be returned. false=a sample is returned
	void evaluationType(bool forward,bool evalMean){
		m_forward = forward;
		m_evalMean = evalMean;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	///\brief Passes information through/samples from an RBM in a forward or backward way. 
	///
	///Eval performs its operation based on the given evaluation type.
	///There are two ways to pass data through an RBM: either forward, setting the states of the
	///visible neurons and sample the hidden states or backwards, where the state of the hidden is fixed and the visible
	///are sampled. 
	///Instead of the state of the hidden/visible, one often wants the mean of the state \f$ E_{p(h|v)}\left(h\right)\f$. 
	///By default, the RBM uses the forward evaluation and returns the mean of the state,
	///but other evaluation modes can be set by evaluationType().
	///
	///@param patterns the batch of (visible or hidden) inputs
	///@param outputs the batch of (visible or hidden) outputs 
	void eval(BatchInputType const& patterns,BatchOutputType& outputs)const{
		if(m_forward){
			evalForward(patterns,outputs);
		}
		else{
			evalBackward(patterns,outputs);
		}
	}


	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}
	using base_type::eval;
	
	
	///\brief Returns the number of hidden Neurons.
	std::size_t numberOfHN()const{
		return m_hiddenNeurons.size();
	}
	///\brief Returns the number of visible Neurons.
	std::size_t numberOfVN()const{
		return m_visibleNeurons.size();
	}
	
	/// \brief Reads the network from an archive.
	void read(InArchive& archive){
		archive >> m_weightMatrix;
		archive >> m_hiddenNeurons;
		archive >> m_visibleNeurons;
		
		//serialization of the rng is a bit...complex
		//let's hope that we can remove this hack one time. But we really can't ignore the state of the rng.
		std::string str;
		archive>> str;
		std::stringstream stream(str);
		stream>> *mpe_rng;
	}

	/// \brief Writes the network to an archive.
	void write(OutArchive& archive) const{
		archive << m_weightMatrix;
		archive << m_hiddenNeurons;
		archive << m_visibleNeurons;
		
		std::stringstream stream;
		stream <<*mpe_rng;
		std::string str = stream.str();
		archive <<str;
	}

};

}

#endif
