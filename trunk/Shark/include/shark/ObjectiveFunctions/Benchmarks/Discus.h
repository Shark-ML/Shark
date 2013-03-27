/**
 *
 * \brief Convex quadratic benchmark function.
 * \date 2010-2011
 * \par Copyright (c):
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 * <BR><HR>
 * This file is part of Shark. This library is free software;
 * you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software
 * Foundation; either version 3, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#ifndef SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_DISCUS_H
#define SHARK_OBJECTIVEFUNCTIONS_BENCHMARKS_DISCUS_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Rng/GlobalRng.h>

namespace shark {
/**
 * \brief Convex quadratic benchmark function.
 */
struct Discus : public SingleObjectiveFunction {

	Discus(unsigned int numberOfVariables = 5,double alpha = 1.E-3) : m_alpha(alpha) {
		m_features |= CAN_PROPOSE_STARTING_POINT;
		m_numberOfVariables = numberOfVariables;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Discus"; }

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
		double sum = m_alpha * sqr(p(0));
		for (unsigned int i = 1; i < p.size(); i++)
			sum += sqr(p(i));

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

ANNOUNCE_SINGLE_OBJECTIVE_FUNCTION(Discus, shark::soo::RealValuedObjectiveFunctionFactory);
}

#endif
