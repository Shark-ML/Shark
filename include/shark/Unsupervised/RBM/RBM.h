/*
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
*/
#ifndef SHARK_UNSUPERVISED_RBM_RBM_H
#define SHARK_UNSUPERVISED_RBM_RBM_H

#include <shark/Models/AbstractModel.h>
#include <shark/LinAlg/Base.h>

#include <sstream>
#include <boost/serialization/string.hpp>
namespace shark{

///\brief stub for the RBM class. at the moment it is just a holder of the parameter set and the Energy.
template<class EnergyType, class RngT>
class RBM : public AbstractModel<typename EnergyType::VectorType,typename EnergyType::VectorType>{
public:
	typedef EnergyType Energy;
	typedef typename Energy::Structure Structure;
	typedef RngT RngType;
	typedef typename Energy::VectorType VectorType; 
	typedef AbstractModel<VectorType,VectorType> base_type;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;
	
private:
	Structure m_structure;
	RngType* mpe_rng;
	bool m_forward;
	bool m_evalMean;

	///\brief Evaluates the input by propagating the visible input to the hidden neurons.
	void evalForward(BatchInputType const& state,BatchOutputType& output)const{
		Energy energy(&m_structure);
		typedef Batch<typename Energy::HiddenInput> InputTraits;
		typedef Batch<typename Energy::HiddenStatistics> StatisticsTraits;
		
		std::size_t batchSize=state.size1();

		typename StatisticsTraits::type statisticsBatch(batchSize,numberOfHN());
		typename InputTraits::type inputBatch(batchSize,numberOfHN());
		output.resize(state.size1(),numberOfHN());
		
		energy.inputHidden(inputBatch,state);
		hiddenNeurons().sufficientStatistics(inputBatch,statisticsBatch,RealScalarVector(batchSize,1.0));

		if(m_evalMean){
			noalias(output) = hiddenNeurons().mean(statisticsBatch);
		}
		else{
			hiddenNeurons().sample(statisticsBatch,output,*mpe_rng);
		}
	}
	///\brief Evaluates the input by propagating the hidden input to the visible neurons.
	void evalBackward(BatchInputType const& state,BatchOutputType& output)const{
		Energy energy(&m_structure);
		typedef Batch<typename Energy::VisibleInput> InputTraits;
		typedef Batch<typename Energy::VisibleStatistics> StatisticsTraits;
		
		std::size_t batchSize = state.size1();
		
		typename StatisticsTraits::type statisticsBatch(batchSize,numberOfVN());
		typename InputTraits::type inputBatch(batchSize,numberOfVN());
		output.resize(batchSize,numberOfVN());
		
		energy.inputVisible(inputBatch,state);
		visibleNeurons().sufficientStatistics(inputBatch,statisticsBatch,RealScalarVector(batchSize,1.0));
		
		if(m_evalMean){
			noalias(output) = visibleNeurons().mean(statisticsBatch);
		}
		else{
			visibleNeurons().sample(statisticsBatch,output,*mpe_rng);
		}
	}
public:
	RBM(RngType& rng):mpe_rng(&rng),m_forward(true),m_evalMean(true){
		this->m_name="RBM";
	}
	
	///\brief Returns the total number of parameters of the model.
	std::size_t numberOfParameters()const {
		return m_structure.numberOfParameters();
	}
	
	///\brief Returns the parameters of the Model as parameter vector.
	RealVector parameterVector () const {
		return m_structure.parameterVector();
	};

	///\brief Sets the parameters of the model.
 	void setParameterVector(const RealVector& newParameters) {
		m_structure.setParameterVector(newParameters);
 	}
	
	///\brief Configures the structure.
	void configure( const PropertyTree & node ){
		m_structure.configure(node);
	}

	///\brief Returns the internal structure of the RBM parameters.
	const Structure& structure()const{
		return m_structure;
	}
	///\brief Returns the internal structure of the RBM parameters.
	Structure& structure(){
		return m_structure;
	}
	
	///\brief Creates the structure of the RBM.
	///
	///@param hiddenNeurons number of hidden neurons.
	///@param visibleNeurons number of visible neurons.
	void setStructure(std::size_t hiddenNeurons,std::size_t visibleNeurons){
		structure().setStructure(hiddenNeurons,visibleNeurons);
	}
	
	///\brief Returns the layer of hidden neurons.
	const typename Energy::HiddenType& hiddenNeurons()const{
		return m_structure.hiddenNeurons();
	}
	///\brief Returns the layer of hidden neurons.
	typename Energy::HiddenType& hiddenNeurons(){
		return m_structure.hiddenNeurons();
	}
	///\brief Returns the layer of visible neurons.
	typename Energy::VisibleType& visibleNeurons(){
		return m_structure.visibleNeurons();
	}
	///\brief Returns the layer of visible neurons.
	const typename Energy::VisibleType& visibleNeurons()const{
		return m_structure.visibleNeurons();
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
	///@param forward whether the forward view should be used false=backwards
	///@param evalMean whether the mean state should be returned. false=a sample is returned
	void evaluationType(bool forward,bool evalMean){
		m_forward = forward;
		m_evalMean = evalMean;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	void eval(BatchInputType const& patterns,BatchOutputType& outputs)const{
		if(m_forward){
			evalForward(patterns,outputs);
		}
		else{
			evalBackward(patterns,outputs);
		}
	}
	void eval(BatchInputType const& patterns,BatchOutputType& outputs, State& state)const{
		eval(patterns,outputs);
	}
	using base_type::eval;
	
	
	///\brief Returns the number of hidden Neurons.
	std::size_t numberOfHN()const{
		return m_structure.numberOfHN();
	}
	///\brief Returns the number of visible Neurons.
	std::size_t numberOfVN()const{
		return m_structure.numberOfVN();
	}
	
	/// \brief Reads the network from an archive.
	void read(InArchive& archive){
		archive >> m_structure;
		
		//serialization of the rng is a bit...complex
		//let's hope that we can remove this hack one time. But we really can't ignore the state of the rng.
		std::string str;
		archive>> str;
		std::stringstream stream(str);
		stream>> *mpe_rng;
	}

	/// \brief Writes the network to an archive.
	void write(OutArchive& archive) const{
		archive << m_structure;
		
		std::stringstream stream;
		stream <<*mpe_rng;
		std::string str = stream.str();
		archive <<str;
	}

};

}

#endif
