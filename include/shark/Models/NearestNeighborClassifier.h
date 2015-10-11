//===========================================================================
/*!
 * 
 *
 * \brief       Nearest neighbor classification
 * 
 * 
 *
 * \author      T. Glasmachers, O. Krause
 * \date        2012
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
//===========================================================================

#ifndef SHARK_MODELS_NEARESTNEIGHBORCLASSIFIER_H
#define SHARK_MODELS_NEARESTNEIGHBORCLASSIFIER_H

#include <shark/Models/AbstractModel.h>
#include <shark/Algorithms/NearestNeighbors/AbstractNearestNeighbors.h>
#include <algorithm>
namespace shark {


///
/// \brief Nearest Neighbor Classifier.
///
/// \par
/// The NearestNeighborClassifier predicts a class label
/// according to a local majority decision among its k
/// nearest neighbors. It is not specified how ties are
/// broken.
///
/// This model requires the use of one of sharks nearest neighhbor Algorithms.
/// \see AbstractNearestNeighbors
template <class InputType>
class NearestNeighborClassifier : public AbstractModel<InputType, unsigned int>
{
public:
	typedef AbstractNearestNeighbors<InputType,unsigned int> NearestNeighbors;
	typedef AbstractModel<InputType, unsigned int> base_type;
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::BatchOutputType BatchOutputType;

	/// \brief Type of distance-based weights.
	enum DistanceWeights
	{
		UNIFORM,                ///< uniform (= no) distance-based weights
		ONE_OVER_DISTANCE,      ///< weight each neighbor's label with 1/distance
	};

	///\brief Constructor
	///
	/// \param algorithm the used algorithm for nearst neighbor search
	/// \param neighbors: number of neighbors
	NearestNeighborClassifier(NearestNeighbors const* algorithm, unsigned int neighbors = 3)
	: m_algorithm(algorithm)
	, m_classes(numberOfClasses(algorithm->dataset()))
	, m_neighbors(neighbors)
	, m_distanceWeights(UNIFORM)
	{ }

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NearestNeighborClassifier"; }


	/// return the number of neighbors
	unsigned int neighbors() const{
		return m_neighbors;
	}

	/// set the number of neighbors
	void setNeighbors(unsigned int neighbors){
		m_neighbors=neighbors;
	}

	/// query the way distances enter as weights
	DistanceWeights getDistanceWeightType() const
	{ return m_distanceWeights; }

	/// set the way distances enter as weights
	void setDistanceWeightType(DistanceWeights dw)
	{ m_distanceWeights = dw; }

	/// get internal parameters of the model
	virtual RealVector parameterVector() const{
		RealVector parameters(1);
		parameters(0) = m_neighbors;
		return parameters;
	}

	/// set internal parameters of the model
	virtual void setParameterVector(RealVector const& newParameters){
		SHARK_CHECK(newParameters.size() == 1,
			"[SoftNearestNeighborClassifier::setParameterVector] invalid number of parameters");
		//~ SHARK_CHECK((unsigned int)newParameters(0) == newParameters(0) && newParameters(0) >= 1.0,
			//~ "[SoftNearestNeighborClassifier::setParameterVector] invalid number of neighbors");
		m_neighbors = (unsigned int)newParameters(0);
	}

	/// return the size of the parameter vector
	virtual std::size_t numberOfParameters() const{
		return 1;
	}
	
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	using base_type::eval;

	/// soft k-nearest-neighbor prediction
	void eval(BatchInputType const& patterns, BatchOutputType& output, State& state)const{
		std::size_t numPatterns = shark::size(patterns);
		std::vector<typename NearestNeighbors::DistancePair> neighbors = m_algorithm->getNeighbors(patterns,m_neighbors);

		output.resize(numPatterns);
		output.clear();

		for(std::size_t p = 0; p != numPatterns;++p){
			std::vector<double> histogram(m_classes, 0.0);
			for ( std::size_t k = 0; k != m_neighbors; ++k){
				if (m_distanceWeights == UNIFORM) histogram[neighbors[p*m_neighbors+k].value]++;
				else
				{
					double d = neighbors[p*m_neighbors+k].key;
					if (d < 1e-100) histogram[neighbors[p*m_neighbors+k].value] += 1e100;
					else histogram[neighbors[p*m_neighbors+k].value] += 1.0 / d;
				}
			}
			output(p) = std::max_element(histogram.begin(),histogram.end()) - histogram.begin();
		}
	}

	/// from ISerializable, reads a model from an archive
	void read(InArchive& archive){
		archive & m_neighbors;
		archive & m_classes;
	}

	/// from ISerializable, writes a model to an archive
	void write(OutArchive& archive) const{
		archive & m_neighbors;
		archive & m_classes;
	}

protected:
	NearestNeighbors const* m_algorithm;

	/// number of classes
	unsigned int m_classes;

	/// number of neighbors to be taken into account
	unsigned int m_neighbors;

	/// type of distance-based weights computation
	DistanceWeights m_distanceWeights;
};



}
#endif
