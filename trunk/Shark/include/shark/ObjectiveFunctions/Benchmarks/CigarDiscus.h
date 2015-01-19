/*!
 * 
 *
 * \brief       Convex quadratic benchmark function.
 * 
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGARDISCUS_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARK_CIGARDISCUS_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/**
 * \brief Convex quadratic benchmark function.
 */
class CigarDiscus : public SingleObjectiveFunction {
public:

	CigarDiscus(unsigned int numberOfVariables = 5,double alpha=1E-3) : m_alpha(alpha) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "CigarDiscus"; }

	std::size_t numberOfVariables()const{
		return m_numberOfVariables;
	}
	
	bool hasScalableDimensionality()const{
		return true;
	}
	
	/// \brief Adjusts the number of variables if the function is scalable.
	/// \param [in] numberOfVariables The new dimension.
	void setNumberOfVariables( std::size_t numberOfVariables ){
		m_numberOfVariables = numberOfVariables;
	}

	void configure(const PropertyTree &node) {
		m_alpha = node.get("alpha", 1E-3);
	}

	void proposeStartingPoint(SearchPointType &x) const {
		x.resize(m_numberOfVariables);

		for (unsigned int i = 0; i < x.size(); i++) {
			x(i) = Rng::uni(0, 1);
		}
	}

	double eval(const SearchPointType &p) const {
		m_evaluationCounter++;

		double sum = m_alpha * sqr(p(0)) + sqr(p(p.size() - 1));
		double alpha = ::sqrt(m_alpha);
		for (unsigned int i = 1; i < p.size()-1; i++)
			sum += alpha * sqr(p(i));

		return sum;
	}

	double alpha() const {
		return m_alpha;
	}

	void setAlpha(double alpha) {
		m_alpha = alpha;
	}

private:
	double m_alpha;
	std::size_t m_numberOfVariables;
};
}

#endif
