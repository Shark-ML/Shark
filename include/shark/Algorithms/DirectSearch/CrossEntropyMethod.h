//===========================================================================
/*!
 *
 * \brief       Implements the Cross Entropy Algorithm.
 * 
 * Christophe Thiery, Bruno Scherrer. Improvements on Learning Tetris with Cross Entropy.
 * International Computer Games Association Journal, ICGA, 2009, 32. <inria-00418930>
 * 
 *
 * \author      Jens Holm, Mathias Petræus and Mark Wulff
 * \date        January 2016
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


#ifndef SHARK_ALGORITHMS_DIRECT_SEARCH_CROSSENTROPY_H
#define SHARK_ALGORITHMS_DIRECT_SEARCH_CROSSENTROPY_H

#include <shark/Core/DLLSupport.h>
#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/Random.h>
#include <shark/Algorithms/DirectSearch/Individual.h>

#include <boost/shared_ptr.hpp>

namespace shark {

/// \brief Implements the Cross Entropy Method.
///
///  This class implements the noisy cross entropy method 
///  as descibed in the following article.
///
///  Christophe Thiery, Bruno Scherrer. Improvements on Learning Tetris with Cross Entropy.
///  International Computer Games Association Journal, ICGA, 2009, 32. <inria-00418930>
///
///  The algorithm aims to minimize an objective function through stochastic search.
///  It works iteratively until a certain stopping criteria is met. At each 
///  iteration, it samples a number of vectors from a Gaussian distribution
///  and evaluates each of these against the supplied objective function.
///  Based on the return value from the objective function, a subset of the 
///  the best ranked vectors are chosen to update the search parameters 
///  of the next generation. 
///
///  The mean of the Gaussian distribution is set to the centroid of the best ranked 
///  vectors, and the variance is set to the variance of the best ranked 
///  vectors in each individual dimension.
/// \ingroup singledirect
class CrossEntropyMethod : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:

	/// \brief Interface class for noise type.
	class INoiseType {
	public:
		virtual double noiseValue (int t) const { return 0.0; };
		virtual std::string name() const { return std::string("Default noise of 0"); }
	};

	/// \brief Smart pointer for noise type.
	typedef boost::shared_ptr<INoiseType> StrongNoisePtr;

	/// \brief Constant noise term z_t = noise.
	class ConstantNoise : public INoiseType {
	public:
		ConstantNoise ( double noise ) { m_noise = noise; };
		virtual double noiseValue (int t) const { return std::max(m_noise, 0.0); }
		virtual std::string name() const {
			std::stringstream ss;
			ss << "z(t) = " << m_noise;
			return std::string(ss.str());
		}
	private:
		double m_noise;
	};

	/// \brief Linear noise term z_t = a + t / b.
	class LinearNoise : public INoiseType {
	public:
		LinearNoise ( double a, double b ) { m_a = a; m_b = b; };
		virtual double noiseValue (int t) const { return std::max(m_a + (t * m_b), 0.0); }
		virtual std::string name() const {
			std::stringstream ss;
			std::string sign = (m_b < 0.0 ? " - " : " + ");
			ss << "z(t) = " << m_a << sign << "t * " << std::abs(m_b);
			return std::string(ss.str());
		}
	private:
		double m_a, m_b;
	};
	

	/// \brief Default c'tor.
	SHARK_EXPORT_SYMBOL CrossEntropyMethod();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Cross Entropy Method"; }

	/// \brief Sets default value for Population size.
	SHARK_EXPORT_SYMBOL static unsigned suggestPopulationSize(  ) ;

	/// \brief Calculates selection size for the supplied population size.
	SHARK_EXPORT_SYMBOL static unsigned suggestSelectionSize( unsigned int populationSize ) ;

	void read( InArchive & archive );
	void write( OutArchive & archive ) const;

	using AbstractSingleObjectiveOptimizer<RealVector >::init;

	/// \brief Initializes the algorithm for the supplied objective function and the initial mean p.
	SHARK_EXPORT_SYMBOL void init( ObjectiveFunctionType const& function, SearchPointType const& p);

	/// \brief Initializes the algorithm for the supplied objective function.
	SHARK_EXPORT_SYMBOL void init(
		ObjectiveFunctionType const& function,
		SearchPointType const& initialSearchPoint,
		unsigned int populationSize,
		unsigned int selectionSize,
		RealVector initialSigma
	);

	/// \brief Executes one iteration of the algorithm.
	SHARK_EXPORT_SYMBOL void step(ObjectiveFunctionType const& function);

	/// \brief Access the current variance.
	RealVector const& variance() const {
		return m_variance;
	}

	/// \brief Set the variance to a vector.
	void setVariance(RealVector variance) {
		assert(variance.size() == m_variance.size());
		m_variance = variance;
	}

	/// \brief Set all variance values.
	void setVariance(double variance){
			m_variance = blas::repeat(variance,m_variance.size());
	}

	/// \brief Access the current population mean.
	RealVector const& mean() const {
		return m_mean;
	}

	/// \brief Returns the size of the parent population.
	unsigned int selectionSize() const {
		return m_selectionSize;
	}

	/// \brief Returns a mutable reference to the size of the parent population.
	unsigned int& selectionSize(){
		return m_selectionSize;
	}

	/// \brief Returns a immutable reference to the size of the population.
	unsigned int populationSize()const{
		return m_populationSize;
	}

	/// \brief Returns a mutable reference to the size of the population.
	unsigned int & populationSize(){
		return m_populationSize;
	}

	/// \brief Set the noise type from a raw pointer.
	void setNoiseType( INoiseType* noiseType ) {
		m_noise.reset();
		m_noise = boost::shared_ptr<INoiseType> (noiseType);
	}

	/// \brief Get an immutable reference to Noise Type.
	const INoiseType &getNoiseType( ) const {
		return *m_noise.get();
	}


protected:
	typedef Individual<RealVector, double, RealVector> IndividualType;
	/// \brief Updates the strategy parameters based on the supplied parent population.
	SHARK_EXPORT_SYMBOL void updateStrategyParameters( std::vector< IndividualType > const& parents ) ;

	std::size_t m_numberOfVariables;///< Stores the dimensionality of the search space.
	unsigned int m_selectionSize;///< Number of vectors chosen when updating distribution parameters.
	unsigned int m_populationSize;///< Number of vectors sampled in a generation.

	RealVector m_variance;///< Variance for sample parameters.

	
	RealVector m_mean;///< The mean of the population.

	unsigned m_counter;///< Counter for generations.

	StrongNoisePtr m_noise;///< Noise type to apply in the update of distribution parameters.
};
}

#endif
