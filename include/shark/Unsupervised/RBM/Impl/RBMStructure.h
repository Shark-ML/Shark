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
#ifndef SHARK_UNSUPERVISED_RBM_RBMSTRUCTURE_H
#define SHARK_UNSUPERVISED_RBM_RBMSTRUCTURE_H

#include <shark/Core/IParameterizable.h>
#include <shark/Core/ISerializable.h>
#include <shark/Core/IConfigurable.h>
#include <shark/Rng/GlobalRng.h>
#include <shark/LinAlg/Base.h>
#include <shark/LinAlg/BLAS/Initialize.h>
#include <boost/array.hpp>
namespace shark{

/// \brief The structure of an RBM is defined by the neuron type of its layers and its interaction parameters.
template<class HiddenType, class VisibleType, class VectorT, std::size_t hiddenFeatures, std::size_t visibleFeatures>
class RBMStructure : public IParameterizable,public ISerializable,public IConfigurable {
public:
	typedef VectorT VectorType;
	typedef typename VectorMatrixTraits<VectorType>::MatrixType MatrixType;
protected:

	///Matrices that identify the weights associated with the connections between the visible and the hidden neurons 
	///(In the standard case just one single matrix)
	boost::array<MatrixType,hiddenFeatures * visibleFeatures> m_weightMatrix;

	///The layer of hidden Neurons
	HiddenType m_hiddenNeurons;

	///The Layer of visible Neurons
	VisibleType m_visibleNeurons;

public:
	
	/// \brief Reads the network from an archive.
	void read(InArchive& archive)
	{
		archive >> m_weightMatrix;
		archive >> m_hiddenNeurons;
		archive >> m_visibleNeurons;
	}

	/// \brief Writes the network to an archive.
	void write(OutArchive& archive) const
	{
		archive << m_weightMatrix;
		archive << m_hiddenNeurons;
		archive << m_visibleNeurons;
	}

	/// \brief Configures the network.
	///
	///two properties must be avaiable:
	///numberOfHN :number of hidden neurons
	///numberOfVN :number of visible neurons
	void configure( const PropertyTree & node ){
		size_t numberOfHN = node.get<unsigned int>("numberOfHN");
		size_t numberOfVN = node.get<unsigned int>("numberOfVN");
		setStructure(numberOfVN,numberOfHN);
	}


	/// \brief Returns the total number of parameters of the model.
	size_t numberOfParameters()const {
		size_t parameters = numberOfVN()*numberOfHN();
		parameters += m_hiddenNeurons.numberOfParameters();
		parameters += m_visibleNeurons.numberOfParameters();
		return parameters;
	}

	/// \brief Returns the parameters of the RBM as ParameterVector.
	RealVector parameterVector () const {
		RealVector ret(numberOfParameters());
		init(ret) << matrixSet(m_weightMatrix),parameters(m_hiddenNeurons),parameters(m_visibleNeurons);

		return ret;
	};

	/// \brief Sets the parameters of the RBM.
 	void setParameterVector(const RealVector& newParameters) {
		init(newParameters) >> matrixSet(m_weightMatrix),parameters(m_hiddenNeurons),parameters(m_visibleNeurons);
 	}

	/// \brief Creates an RBM with a specified number of visible and hidden units.
	///
	///@param numOfVN number of visible units
	///@param numOfHN number of hidden units
	void setStructure(size_t numOfVN,size_t numOfHN) {
		for(std::size_t i = 0; i != hiddenFeatures* visibleFeatures; ++i){
			m_weightMatrix[i].resize(numOfHN,numOfVN);
			m_weightMatrix[i].clear();
		}
		
		m_hiddenNeurons.resize(numOfHN);
		m_visibleNeurons.resize(numOfVN);
	}

	/// \brief Returns the weight matrix corresponding to the i-th phi-functions
	/// of the hidden an the j-th phi-function of the visible variables.
	///
	/// @param i index of the hidden phi-function
	/// @param j index of the visible phi-function
	const RealMatrix& weightMatrix(std::size_t i, std::size_t j) const {
		return m_weightMatrix[i*visibleFeatures+j];
	};

	/// \brief Returns the weight matrix corresponding to the i-th phi-functions
	/// of the hidden an the j-th phi-function of the visible variables.
	///
	/// @param i index of the hidden phi-function
	/// @param j index of the visible phi-function
	RealMatrix& weightMatrix(std::size_t i, std::size_t j){
		return m_weightMatrix[i*visibleFeatures+j];
	};

	/// \brief Returns the number of phi-functions of the hidden neuron layer.
	std::size_t numberOfHiddenFeatures(){
		return hiddenFeatures;
	}

	/// \brief Returns the number of phi-functions of the hidden neuron layer.
	std::size_t numberOfVisibleFeatures(){
		return visibleFeatures;
	}

	/// \brief Returns the layer of hidden neurons.
	HiddenType& hiddenNeurons(){
		return m_hiddenNeurons;
	}

	/// \brief Returns the layer of hidden neurons.
	const HiddenType& hiddenNeurons()const{
		return m_hiddenNeurons;
	}

	/// \brief Returns the layer of visible neurons.
	VisibleType& visibleNeurons(){
		return m_visibleNeurons;
	}

	/// \brief Returns the layer of visible neurons.
	const VisibleType& visibleNeurons()const{
		return m_visibleNeurons;
	}
	
	/// \brief Returns the number of hidden neurons.
	size_t numberOfHN() const {
		return m_hiddenNeurons.size();
	}

	/// \brief Returns the number of visible neurons.
	size_t numberOfVN() const {
		return m_visibleNeurons.size();
	}


};

}

#endif
