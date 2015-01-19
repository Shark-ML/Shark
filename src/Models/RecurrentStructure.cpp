/*!
 * 
 *
 * \brief       Base class for Recurrent Neural Networks
 * 
 *
 * \author      -
 * \date        -
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


#include <shark/Models/RecurrentStructure.h>
#include <shark/Core/Math.h>
#include <boost/math/special_functions/sign.hpp>

#include <map>

using namespace std;
using namespace shark;


RecurrentStructure::RecurrentStructure()
:m_numberOfNeurons(0),m_inputNeurons(0),m_outputNeurons(0),m_numberOfParameters(0)
{}

RealVector RecurrentStructure::parameterVector() const
{
	RealVector parameters(m_numberOfParameters);
	std::size_t param =0;
	for (size_t i = 0; i != m_numberOfNeurons; ++i){
		for (size_t j = 0; j != m_numberOfUnits; ++j){
			if(!m_connectionMatrix(i,j))continue;
			parameters(param)=m_weights(i,j);
			++param;
		}
	}
	//sanity check
	SIZE_CHECK(param == m_numberOfParameters);
	return parameters;
}
void RecurrentStructure::setParameterVector(RealVector const& newParameters)
{
	SIZE_CHECK(newParameters.size() == m_numberOfParameters);
	std::size_t param =0;
	for (std::size_t i = 0; i != m_numberOfNeurons; ++i){
		for (std::size_t j = 0; j != m_numberOfUnits; ++j){
			if(!m_connectionMatrix(i,j))continue;
			m_weights(i,j) = newParameters(param);
			++param;
		}
	}
	//sanity check
	SIZE_CHECK(param == m_numberOfParameters);
}

void RecurrentStructure::setWeights(RealMatrix const& weights){
	m_weights=weights;
}

void RecurrentStructure::setStructure(std::size_t inputs, std::size_t outputs, const IntMatrix& connections, SigmoidType sigmoidType){
	SIZE_CHECK(connections.size1()==connections.size2()-inputs-1);
	SIZE_CHECK(connections.size2()>=inputs+outputs);
	m_sigmoidType = sigmoidType;

	m_inputNeurons = inputs;
	m_outputNeurons = outputs;
	m_numberOfNeurons = connections.size1();
	m_numberOfUnits =  connections.size2();//with bias
	m_bias = m_inputNeurons;//index of bias neuron

	m_weights.resize(m_numberOfNeurons, m_numberOfUnits);
	m_weights.clear();
	m_connectionMatrix = connections;

	m_numberOfParameters = 0;
	for (size_t i = 0;i != m_numberOfNeurons;i++){
		for (size_t j = 0;j != m_numberOfUnits;j++){
			if (m_connectionMatrix(i, j))
				m_numberOfParameters++;
		}
	}
}
void RecurrentStructure::setStructure(std::size_t in, std::size_t hidden, std::size_t out, bool bias, SigmoidType sigmoidType)
{
	// Calculate total number of neurons from the
	// number of neurons per layer:
	std::size_t n = in+hidden+out;
	IntMatrix connections = IntScalarMatrix(hidden+out,n+1,1);

	if(!bias){
		column(connections,in).clear();
	}
	setStructure(in,out,connections,sigmoidType);
}

void RecurrentStructure::read( InArchive & archive ){
	archive >> m_weights;
	archive >> m_connectionMatrix;
	archive >> m_inputNeurons;
	archive >> m_outputNeurons;
	archive >> m_numberOfParameters;
	
	m_numberOfNeurons = m_connectionMatrix.size1();
	m_numberOfUnits = m_connectionMatrix.size2();
	m_bias = m_inputNeurons;
}

void RecurrentStructure::write( OutArchive & archive ) const{
	archive << m_weights;
	archive << m_connectionMatrix;
	archive << m_inputNeurons;
	archive << m_outputNeurons;
	archive << m_numberOfParameters;
}
void RecurrentStructure::configure( const PropertyTree & node ){
	try{
		size_t inputNeurons = node.get<size_t>("inputs");
		size_t hiddenNeurons = node.get<size_t>("hidden");
		size_t outputNeurons = node.get<size_t>("outputs");
		bool bias = node.get("bias",true);

		string sigmoidTypeS =node.get<string>("sigmoidType","Logistic");
		map<string,SigmoidType> types;
		types.insert(make_pair("tanh",Tanh));
		types.insert(make_pair("logistic",Logistic));
		types.insert(make_pair("linear",Linear));
		types.insert(make_pair("fastSigmoid",FastSigmoid));

		if(types.find(sigmoidTypeS)==types.end() )
			SHARKEXCEPTION("[FFNet::configure] unknown type of Sigmoid");
		SigmoidType sigmoidType = types.find(sigmoidTypeS)->second;

		setStructure(inputNeurons,hiddenNeurons,outputNeurons,bias,sigmoidType);


	}
	catch(boost::property_tree::ptree_error&){
		SHARKEXCEPTION("[RecurrentStructure::configure] wrong data format or missing data");
	}
}
double RecurrentStructure::neuron(double a) {
	switch(m_sigmoidType){
		case Tanh:
			return std::tanh(a);
		case Logistic:
			return 1.0 / (1.0 + std::exp(-a));
		case Linear:
			return a;
		case FastSigmoid:
			return a/(1+std::abs(a));
	}
	return 0;
}
double RecurrentStructure::neuronDerivative(double ga) {
	switch(m_sigmoidType){
		case Tanh:
			return 1 - ga * ga;;
		case Logistic:
			return ga *(1 - ga);
		case Linear:
			return 1;
		case FastSigmoid:
			return sqr(1 - boost::math::sign(ga) * ga);
	}
	return 0;
}
