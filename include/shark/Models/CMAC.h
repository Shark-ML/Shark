/*!
 * 
 *
 * \brief       -
 *
 * \author      O. Krause
 * \date        2010-01-01
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

#ifndef SHARK_ML_MODEL_CMAC_H
#define SHARK_ML_MODEL_CMAC_H

#include <shark/Core/DLLSupport.h>
#include <shark/Models/AbstractModel.h>
#include <shark/Rng/GlobalRng.h>
#include <vector>

namespace shark{

//!
//! \brief The CMACMap class represents a linear combination of piecewise constant functions
//!
//! when a point is fed into the CMAC, it is first mapped into a vector of binary features.
//! For this purpose the inputspace is divided into several tilings. Every tiling produces a bitstring where an element
//! is 1 if the point lies inside the tile, 0 otherwise. The concatenation of all tilings forms the feature vector which is then fed
//! into a linear function.
//! Usually the CMAC is only good for low dimensional input data since the size of the featurevector grows exponentially with the
//! number of dimensions.
//!
class CMACMap :public AbstractModel<RealVector,RealVector>{
protected:
	///offset of the position of every tiling
	RealMatrix m_offset;

	///coordinate offset for every dimension in the Array
	std::vector<std::size_t> m_dimOffset;

	///lower bound and tileWidth for every Dimension
	RealMatrix m_tileBounds;

	///number of tilings
	std::size_t m_tilings;
	std::size_t m_parametersPerTiling;

	std::size_t m_inputSize;
	std::size_t m_outputSize;

	///The parameters of the model
	RealVector m_parameters;

	///calculates the index in the parameter vector for the activated feature in the tiling
	SHARK_EXPORT_SYMBOL std::size_t getArrayIndexForTiling(std::size_t indexOfTiling,RealVector const& point)const;
	///returns an index in the parameter array for each activated feature
	SHARK_EXPORT_SYMBOL std::vector<std::size_t> getIndizes(ConstRealMatrixRow const& point)const;
public:
	///\brief construct the CMAC
	SHARK_EXPORT_SYMBOL CMACMap();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CMACMap"; }

	///\brief initializes the structure of the cmac. it uses the same lower and upper bound for every input dimension. default is [0,1]
	///
	///\param inputs number of input dimensions
	///\param outputs number of output dimensions
	///\param numberOfTilings number of Tilings to be created
	///\param numberOfTiles amount of tiles per dimension
	///\param lower lower bound of input values
	///\param upper upper bound of input values
	///\param randomTiles flag specifying whether distance between tiles is regular or randomized
	SHARK_EXPORT_SYMBOL void setStructure(std::size_t inputs, std::size_t outputs, std::size_t numberOfTilings, std::size_t numberOfTiles, double lower = 0., double upper = 1.,bool randomTiles = false);

	///\brief initializes the structure of the cmac
	///
	///\param inputs number of input dimensions
	///\param outputs number of output dimensions
	///\param numberOfTilings number of Tilings to be created
	///\param numberOfTiles amount of tiles per dimension
	///\param bounds lower and upper bounts for every input dimension. every row consists of (lower,upper)
	///\param randomTiles flag specifying whether distance between tiles is regular or randomized
	SHARK_EXPORT_SYMBOL void setStructure(std::size_t inputs, std::size_t outputs, std::size_t numberOfTilings, std::size_t numberOfTiles, RealMatrix const& bounds,bool randomTiles = false);

	virtual std::size_t inputSize()const
	{
		return m_inputSize;
	}
	virtual std::size_t outputSize()const
	{
		return m_outputSize;
	}

	virtual RealVector parameterVector()const
	{
		return m_parameters;
	}
	virtual void setParameterVector(RealVector const& newParameters)
	{
		SIZE_CHECK(numberOfParameters() == newParameters.size());
		m_parameters=newParameters;
	}
	virtual std::size_t numberOfParameters()const
	{
		return m_parameters.size();
	}

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}
	
	using AbstractModel<RealVector,RealVector>::eval;
	SHARK_EXPORT_SYMBOL void eval(const RealMatrix& patterns,RealMatrix& outputs)const;
	void eval(const RealMatrix& patterns,RealMatrix& outputs, State& state)const{
		eval(patterns,outputs);
	}
	SHARK_EXPORT_SYMBOL void weightedParameterDerivative(
		RealMatrix const& pattern, 
		RealMatrix const& coefficients,  
		State const& state,
		RealVector& gradient)const;

	/// From ISerializable, reads a model from an archive
	SHARK_EXPORT_SYMBOL void read( InArchive & archive );

	/// From ISerializable, writes a model to an archive
	SHARK_EXPORT_SYMBOL void write( OutArchive & archive ) const;
};


}
#endif
