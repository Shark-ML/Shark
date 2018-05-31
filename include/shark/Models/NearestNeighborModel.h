//===========================================================================
/*!
 * 
 *
 * \brief      NEarest neighbor model for classification and regression
 * 
 * 
 *
 * \author      T. Glasmachers, C. Igel, O.Krause
 * \date        2012-2017
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

#ifndef SHARK_MODELS_NEARESTNEIGHBOR_H
#define SHARK_MODELS_NEARESTNEIGHBOR_H


#include <shark/Models/AbstractModel.h>
#include <shark/Models/Classifier.h>
#include <shark/Algorithms/NearestNeighbors/AbstractNearestNeighbors.h>

namespace shark {
	
namespace detail{
template <class InputType, class LabelType>
class BaseNearestNeighbor : public AbstractModel<InputType, RealVector>
{
public:
	typedef AbstractNearestNeighbors<InputType,LabelType> NearestNeighbors;
	typedef AbstractModel<InputType, RealVector> base_type;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// \brief Constructor
	///
	/// \param algorithm the used algorithm for nearest neighbor search
	/// \param neighbors number of neighbors
	BaseNearestNeighbor(NearestNeighbors const* algorithm, std::size_t outputDimensions, unsigned int neighbors = 3)
	: m_algorithm(algorithm)
	, m_outputDimensions(outputDimensions)
	, m_neighbors(neighbors)
	, m_uniform(true)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Internal"; }
	
	Shape inputShape() const{
		return m_algorithm->inputShape();
	}
	Shape outputShape() const{
		return Shape(m_outputDimensions);
	}

	/// return the number of neighbors
	unsigned int neighbors() const{
		return m_neighbors;
	}

	/// set the number of neighbors
	void setNeighbors(unsigned int neighbors){
		m_neighbors=neighbors;
	}
	
	bool uniformWeights() const{
		return m_uniform;
	}
	bool& uniformWeights(){
		return m_uniform;
	}

	/// get internal parameters of the model
	virtual RealVector parameterVector() const{
		RealVector parameters(1);
		parameters(0) = m_neighbors;
		return parameters;
	}

	/// set internal parameters of the model
	virtual void setParameterVector(RealVector const& newParameters){
		SHARK_RUNTIME_CHECK(newParameters.size() == 1,"Invalid number of parameters");
		m_neighbors = (unsigned int)newParameters(0);
	}

	/// return the size of the parameter vector
	virtual std::size_t numberOfParameters() const{
		return 1;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// soft k-nearest-neighbor prediction
	void eval(BatchInputType const& patterns, BatchOutputType& outputs) const {
		std::size_t numPatterns = batchSize(patterns);
		std::vector<typename NearestNeighbors::DistancePair> neighbors = m_algorithm->getNeighbors(patterns, m_neighbors);

		outputs.resize(numPatterns, m_outputDimensions);
		outputs.clear();

		for(std::size_t p = 0; p != numPatterns;++p)
		{
			double wsum = 0.0;
			for ( std::size_t k = 0; k != m_neighbors; ++k)
			{
				double w = 1.0;
				if (!m_uniform){
					double d = neighbors[p*m_neighbors+k].key;
					if (d < 1e-100) w = 1e100;
					else w = 1.0 / d;
				}
				updatePrediction(outputs, p, w, neighbors[p*m_neighbors+k].value);
				wsum += w;
			}
			row(outputs, p) /= wsum;
		}
	}
	
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State&) const {
		eval(patterns,outputs);
	}
	using base_type::eval;

	/// from ISerializable, reads a model from an archive
	void read(InArchive& archive){
		archive & m_neighbors;
		archive & m_outputDimensions;
		archive & m_uniform;
	}

	/// from ISerializable, writes a model to an archive
	void write(OutArchive& archive) const{
		archive & m_neighbors;
		archive & m_outputDimensions;
		archive & m_uniform;
	}
	
private:
	void updatePrediction(RealMatrix& outputs, std::size_t p, double w, unsigned int const label) const{
		outputs(p, label) += w;
	}
	template<class T>
	void updatePrediction(RealMatrix& outputs, std::size_t p, double w, blas::vector<T> const& label)const{
		noalias(row(outputs,p)) += w * label;
	}
	NearestNeighbors const* m_algorithm;

	/// number of classes
	std::size_t m_outputDimensions;

	/// number of neighbors to be taken into account
	unsigned int m_neighbors;

	/// type of distance-based weights computation
	bool m_uniform;
};
}

/// \brief NearestNeighbor model for classification and regression
///
/// The classification, the model predicts a class label
/// according to a local majority decision among its k
/// nearest neighbors. It is not specified how ties are
/// broken.
///
/// For Regression, the (weighted) mean of the k nearest
/// neighbours is computed.
///
/// \ingroup models
template <class InputType, class LabelType>
class NearestNeighborModel: public detail::BaseNearestNeighbor<InputType,LabelType>
{
public:
	typedef AbstractNearestNeighbors<InputType,LabelType> NearestNeighbors;
	typedef detail::BaseNearestNeighbor<InputType,LabelType> base_type;

	/// \brief Type of distance-based weights.
	enum DistanceWeights{
		UNIFORM,                ///< uniform (= no) distance-based weights
		ONE_OVER_DISTANCE,      ///< weight each neighbor's label with 1/distance
	};

	/// \brief Constructor
	///
	/// \param algorithm the used algorithm for nearest neighbor search
	/// \param neighbors number of neighbors
	NearestNeighborModel(NearestNeighbors const* algorithm, unsigned int neighbors = 3)
	: base_type(algorithm, labelDimension(algorithm->dataset()), neighbors)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NearestNeighbor"; }

	/// query the way distances enter as weights
	DistanceWeights getDistanceWeightType() const{
		return this->decisionFunction().uniformWeights() ? UNIFORM : ONE_OVER_DISTANCE;
	}

	/// set the way distances enter as weights
	void setDistanceWeightType(DistanceWeights dw){
		this->decisionFunction().uniformWeights() = (dw == UNIFORM);
	}
};


template <class InputType>
class NearestNeighborModel<InputType, unsigned int>: public Classifier<detail::BaseNearestNeighbor<InputType,unsigned int> >
{
public:
	typedef AbstractNearestNeighbors<InputType,unsigned int> NearestNeighbors;
	typedef Classifier<detail::BaseNearestNeighbor<InputType,unsigned int> > base_type;

	/// \brief Type of distance-based weights.
	enum DistanceWeights{
		UNIFORM,                ///< uniform (= no) distance-based weights
		ONE_OVER_DISTANCE,      ///< weight each neighbor's label with 1/distance
	};

	/// \brief Constructor
	///
	/// \param algorithm the used algorithm for nearest neighbor search
	/// \param neighbors number of neighbors
	NearestNeighborModel(NearestNeighbors const* algorithm, unsigned int neighbors = 3)
	: base_type(detail::BaseNearestNeighbor<InputType,unsigned int>(algorithm, numberOfClasses(algorithm->dataset()), neighbors))
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NearestNeighbor"; }

	/// return the number of neighbors
	unsigned int neighbors() const{
		return this->decisionFunction().neighbors();
	}

	/// set the number of neighbors
	void setNeighbors(unsigned int neighbors){
		this->decisionFunction().setNeighbors(neighbors);
	}

	/// query the way distances enter as weights
	DistanceWeights getDistanceWeightType() const{
		return this->decisionFunction().uniformWeights() ? UNIFORM : ONE_OVER_DISTANCE;
	}

	/// set the way distances enter as weights
	void setDistanceWeightType(DistanceWeights dw){
		this->decisionFunction().uniformWeights() = (dw == UNIFORM);
	}
};


}
#endif
