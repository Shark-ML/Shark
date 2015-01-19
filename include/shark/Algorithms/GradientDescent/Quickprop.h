//===========================================================================
/*!
 * 
 *
 * \brief       Quickprop
 * 
 * 
 *
 * \author      O. Krause
 * \date        2010
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

#ifndef QUICKPROP_H
#define QUICKPROP_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

#include <boost/math/special_functions/sign.hpp>

namespace shark{

/*!
 *  \brief This class offers methods for using the popular heuristic
 *         "Quickprop" optimization algorithm.
 *
 *  The "Quickprop" algorithm sees the weights of a network as if they were
 *  quasi-independant and tries to approximate the error surface, as a
 *  function of each of the weights, by a quadratic polynomial (a parabola).
 *  Then two successive evaluations of the error function and an evaluation
 *  of its gradient follow to determine the coefficients of the
 *  polynomial. At the next step of the iteration, the weight parameter
 *  is moved to the minimum of the parabola. This leads to a calculation
 *  of the update factor for weight no. \f$i,\ i = 1, \dots, n\f$ from
 *  iteration \f$t\f$ to iteration \f$t + 1\f$ as
 *
 *  \f$
 *  \Delta w_i^{(t + 1)} = \frac{g_i^{(t)}}{g_i^{t - 1} - g_i^{t}}
 *                         \Delta w_i^{(t)}
 *  \f$
 *
 *  where \f$g_i^{(t)} = \frac{\partial E}{\partial w_i^{(t)}}\f$. <br>
 *
 *  For further information about this algorithm, please refer to: <br>
 *
 *  S. E. Fahlman, <br>
 *  "Faster-learning variations on back-propagation: an empirical study." <br>
 *  In D. Touretzky, G.E. Hinton and T,J, Sejnowski (Eds.), <br>
 *  "Proceedings of the 1988 Connectionist Models Summer School",
 *  pp. 38-51, San Mateo, CA: Morgan Kaufmann <br>
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class Quickprop : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	Quickprop();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Quickprop"; }

	void configure( const PropertyTree & node );

	void read( InArchive & archive );
	void write( OutArchive & archive ) const;
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	using AbstractSingleObjectiveOptimizer<RealVector >::init;

	double learningRate()const
	{
		return m_learningRate;
	}
	double& learningRate()
	{
		return m_learningRate;
	}
	double maxIncrease()const
	{
		return m_maxIncrease;
	}
	double& maxIncrease()
	{
		return m_maxIncrease;
	}


	//===========================================================================
	/*!
	*  \brief Performs one run of a modified Quickprop algorithm, that
	*         improves the performance.
	*
	*  The error \f$E^{(t)}\f$ of the used network for the current iteration
	*  \f$t\f$ is calculated and the values of the weights \f$w_i,\ i = 1,
	*  \dots, n\f$ are optimized depending on this error.
	*
	*      \param  objectiveFunction    objectivefunction to be optimized
	*/
	void step(const ObjectiveFunctionType& objectiveFunction);
protected:
	ObjectiveFunctionType::FirstOrderDerivative m_derivative;

	//! The update values for all weights.
	RealVector m_deltaw;

	//! The last error gradient.
	RealVector m_oldDerivative;

	size_t m_parameterSize;

	//! The learning rate, set to 1.5 by default.
	double m_learningRate;

	//! Upper limit for the increase factor: No weight adaption may be greater than the factor times the previous adaption, set to 1.75 by default.
	double m_maxIncrease;

};

//===========================================================================
/*!
 *  \brief This class offers methods for using the popular heuristic
 *         "Quickprop" optimization algorithm.
 *
 *  The "Quickprop" algorithm sees the weights of a network as if they were
 *  quasi-independant and tries to approximate the error surface, as a
 *  function of each of the weights, by a quadratic polynomial (a parabola).
 *  Then two successive evaluations of the error function and an evaluation
 *  of its gradient follow to determine the coefficients of the
 *  polynomial. At the next step of the iteration, the weight parameter
 *  is moved to the minimum of the parabola. This leads to a calculation
 *  of the update factor for weight no. \f$i,\ i = 1, \dots, n\f$ from
 *  iteration \f$t\f$ to iteration \f$t + 1\f$ as
 *
 *  \f$
 *  \Delta w_i^{(t + 1)} = \frac{g_i^{(t)}}{g_i^{t - 1} - g_i^{t}}
 *                         \Delta w_i^{(t)}
 *  \f$
 *
 *  where \f$g_i^{(t)} = \frac{\partial E}{\partial w_i^{(t)}}\f$. <br>
 *
 *  For further information about this algorithm, please refer to: <br>
 *
 *  S. E. Fahlman, <br>
 *  "Faster-learning variations on back-propagation: an empirical study." <br>
 *  In D. Touretzky, G.E. Hinton and T,J, Sejnowski (Eds.), <br>
 *  "Proceedings of the 1988 Connectionist Models Summer School",
 *  pp. 38-51, San Mateo, CA: Morgan Kaufmann <br>
 */
 class QuickpropOriginal : public AbstractSingleObjectiveOptimizer<RealVector >
{
public:
	QuickpropOriginal();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Quickprop"; }

	void configure( const PropertyTree & node );

	void read( InArchive & archive );
	void write( OutArchive & archive ) const;

	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	using AbstractSingleObjectiveOptimizer<RealVector >::init;

	double learningRate()const
	{
		return m_learningRate;
	}
	double& learningRate()
	{
		return m_learningRate;
	}
	double maxIncrease()const
	{
		return m_maxIncrease;
	}
	double& maxIncrease()
	{
		return m_maxIncrease;
	}


	//===========================================================================
	/*!
	*  \brief Performs one run of the original Quickprop algorithm.
	*
	*  The error \f$E^{(t)}\f$ of the used network for the current iteration
	*  \f$t\f$ is calculated and the values of the weights \f$w_i,\ i = 1,
	*  \dots, n\f$ are optimized depending on this error.
	*
	*      \param  objectiveFunction    objectivefunction to be optimized
	*/
	void step(const ObjectiveFunctionType& objectiveFunction);
protected:
	ObjectiveFunctionType::FirstOrderDerivative m_derivative;

	//! The update values for all weights.
	RealVector m_deltaw;

	//! The last error gradient.
	RealVector m_oldDerivative;

	size_t m_parameterSize;

	//! The learning rate, set to 1.5 by default.
	double m_learningRate;

	//! Upper limit for the increase factor: No weight adaption may be greater than the factor times the previous adaption, set to 1.75 by default.
	double m_maxIncrease;

};

}
#endif

