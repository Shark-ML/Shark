//===========================================================================
/*!
*
*  \brief Nearest neighbor classification
*
*  \author  T. Glasmachers, O. Krause
*  \date    2012
*
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
	
	///\brief Constructor
	///
	/// \param algorithm the used algorithm for nearst neighbor search
	/// \param neighbors: number of neighbors
	NearestNeighborClassifier(NearestNeighbors const* algorithm, unsigned int neighbors = 3)
	: m_algorithm(algorithm),m_classes(numberOfClasses(algorithm->dataset())), m_neighbors(neighbors){}

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

	/// configures the classifier.
	void configure(PropertyTree const& node){
		m_neighbors = node.get("neighbors", 3);
	}

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
			std::vector<std::size_t> histogram(m_classes,0);
			for ( std::size_t k = 0; k != m_neighbors; ++k){
				histogram[neighbors[p*m_neighbors+k].value]++;
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
};



}
#endif
