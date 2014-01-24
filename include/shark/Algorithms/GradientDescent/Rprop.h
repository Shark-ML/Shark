/*!
 * 
 * \file        Rprop.h
 *
 * \brief       implements different versions of Resilient Backpropagation of error.
 * 
 * 
 * 
 *
 * \author      Oswin Krause
 * \date        2010
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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

#ifndef SHARK_ML_OPTIMIZER_RPROP_H
#define SHARK_ML_OPTIMIZER_RPROP_H


#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>

namespace shark{

/*!
 *  \brief This class offers methods for the usage of the
 *         Resilient-Backpropagation-algorithm without weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independent from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Martin Riedmiller, <br>
 *  "Advanced Supervised Learning in Multi-layer Perceptrons -
 *  From Backpropagation to Adaptive Learning Algorithms". <br>
 *  In "International Journal of Computer Standards and Interfaces", volume 16,
 *  no. 5, 1994, pp. 265-278 <br>
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
class RpropMinus : public AbstractSingleObjectiveOptimizer<VectorSpace<double> >
{
public:
	RpropMinus();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RpropMinus"; }

	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	virtual void init(
		ObjectiveFunctionType const& objectiveFunction, 
		SearchPointType const& startingPoint, 
		double initDelta
	);
	virtual void init(
		ObjectiveFunctionType const& objectiveFunction, 
		SearchPointType const& startingPoint, 
		RealVector const& initDelta
	);
	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;

	void step(const ObjectiveFunctionType& objectiveFunction);
	void configure( const PropertyTree & node );

	virtual void read( InArchive & archive );
	virtual void write( OutArchive & archive ) const;

	//! set decrease factor
	void setEtaMinus(double etaMinus) {
		RANGE_CHECK( etaMinus < 1 );
		RANGE_CHECK( etaMinus > 0 );
		m_decreaseFactor = etaMinus;
	}

	//! set increase factor
	void setEtaPlus(double etaPlus) {
		RANGE_CHECK( etaPlus > 1 );
		m_increaseFactor = etaPlus;
	}

	//! set upper limit on update
	void setMaxDelta(double d) {
		RANGE_CHECK( d > 0 );
		m_maxDelta = d;
	}

	//! set lower limit on update
	void setMinDelta(double d) {
		RANGE_CHECK( d >= 0 );
		m_minDelta = d;
	}

	//! return the maximal step size component
	double maxDelta() const {
		return *std::max_element(m_delta.begin(),m_delta.end());
	}
protected:
	ObjectiveFunctionType::FirstOrderDerivative m_derivative;

	//! The increase factor \f$ \eta^+ \f$, set to 1.2 by default.
	double m_increaseFactor;

	//! The decrease factor \f$ \eta^- \f$, set to 0.5 by default.
	double m_decreaseFactor;

	//! The upper limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 1e100 by default.
	double m_maxDelta;

	//! The lower limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 0.0 by default.
	double m_minDelta;

	size_t m_parameterSize;

	//! The last error gradient.
	RealVector m_oldDerivative;

	//! The absolute update values (increment) for all weights.
	RealVector m_delta;
};

//===========================================================================
/*!
 *  \brief This class offers methods for the usage of the
 *         Resilient-Backpropagation-algorithm with weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independent from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  Furthermore, weight-backtracking will take place to increase the
 *  stability of the method, i.e.
 *  if \f$\frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0\f$ then
 *  \f$\Delta w_i^{(t)} := - \Delta w_i^{(t-1)};
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0\f$, where
 *  the assignment of zero to the partial derivative of the error
 *  leads to a freezing of the increment in the next iteration.
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Martin Riedmiller and Heinrich Braun, <br>
 *  "A Direct Adaptive Method for Faster Backpropagation Learning: The
 *  RPROP Algorithm". <br>
 *  In "Proceedings of the IEEE International Conference on Neural Networks",
 *  pp. 586-591, <br>
 *  Published by IEEE Press in 1993
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
class RpropPlus : public RpropMinus
{
public:
	RpropPlus();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "RpropPlus"; }

	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, double initDelta);
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, const RealVector& initDelta);
	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;

	void step(const ObjectiveFunctionType& objectiveFunction);
	void read( InArchive & archive );
	void write( OutArchive & archive ) const;

protected:
	//! The final update values for all weights.
	RealVector m_deltaw;
};




/*!
 *  \brief This class offers methods for the usage of the improved
 *         Resilient-Backpropagation-algorithm with weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independent from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  Furthermore, weight-backtracking will take place to increase the
 *  stability of the method. In contrast to the original Rprop algorithm
 *  with weight-backtracking (see RpropPlus) this weight-backtracking
 *  is improved by additionally taken the error of the last iteration
 *  \f$t - 1\f$ into account. <br>
 *  The idea of this modification is, that a change of the sign of the
 *  partial derivation \f$\frac{\partial E}{\partial w_i}\f$
 *  only states, that a minimum was skipped and not, whether this step
 *  lead to an approach to the minimum or not. <br>
 *  By using the old error value the improved weight-backtracking only
 *  undoes changes, when the error has increased and only the parameters
 *  \f$w_i\f$ are reset to the old values, where a sign change of
 *  \f$\frac{\partial E}{\partial w_i}\f$ has taken place. <br>
 *  So the new weight-backtracking rule is: <br>
 *
 *  \f$
 *  \mbox{if\ } \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \mbox{\ then} \{
 *  \f$
 *
 *  \f$
 *  \begin{array}{lll}
 *   \Delta w_i^{(t)} = \bigg\{ &
 *   - \Delta w_i^{(t-1)}, & \mbox{if\ } E^{(t)} > E^{(t - 1)} \\
 *   & 0, & otherwise \\
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0
 *  \end{array}
 *  \f$
 *
 *  \f$\}\f$
 *
 *  , where the assignment of zero to the partial derivative of the error
 *  leads to a freezing of the increment in the next iteration. <br>
 *
 *  This modification of the weight backtracking leads to a better
 *  optimization on artifical, paraboloidal error surfaces. <br>
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Christian Igel and Michael H&uuml;sken, <br>
 *  "Empirical Evaluation of the Improved Rprop Learning Algorithm". <br>
 *  In Neurocomputing Journal, 2002, in press <br>
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class IRpropPlus : public RpropPlus
{
public:
	IRpropPlus();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "IRpropPlus"; }

	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint);
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, double initDelta);
	void init(const ObjectiveFunctionType & objectiveFunction, const SearchPointType& startingPoint, const RealVector& initDelta);
	using AbstractSingleObjectiveOptimizer<VectorSpace<double> >::init;

	void step(const ObjectiveFunctionType& objectiveFunction);

	void setDerivativeThreshold(double derivativeThreshold);

	void read( InArchive & archive );
	void write( OutArchive & archive ) const;

protected:
	//! The error of the last iteration.
	double m_oldError;
//! A threshold below which partial derivatives are set to zero
	double m_derivativeThreshold;

};

//===========================================================================
/*!
 *  \brief This class offers methods for the usage of the improved
 *         Resilient-Backpropagation-algorithm without weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates (as the Adaptive Backpropagation algorithm by Silva
 *  and Ameida, please see AdpBP.h for a description of the
 *  working of such an algorithm), that uses increments for the update
 *  of the weights, that are independent from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum. <br>
 *  So, the Rprop algorithm only uses the signs of the partial derivatives
 *  and not the absolute values to adapt the parameters. <br>
 *  Instead of individual learning rates, it uses the parameter
 *  \f$\Delta_i^{(t)}\f$ for weight \f$w_i,\ i = 1, \dots, n\f$ in
 *  iteration \f$t\f$, where the parameter will be adapted before the
 *  change of the weights. <br>
 *  As an improving modification, this algorithm
 *  adapts the "freezing" of the increment in the next iteration as
 *  usually only practiced by the Rprop algorithm with weight-backtracking
 *  (see RpropPlus), i.e. \f$\frac{\partial E^{(t)}}{\partial w_i} := 0\f$.
 *  Tests have shown a far more better optimization when using this
 *  modification. So the new adaptation rule of \f$\Delta\f$ is given
 *  as: <br>
 *
 *  \f$
 *  \Delta_i^{(t)} = \Bigg\{
 *  \begin{array}{ll}
 *  min( \eta^+ \cdot \Delta_i^{(t-1)}, \Delta_{max} ), & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} > 0 \\
 *  max( \eta^- \cdot \Delta_i^{(t-1)}, \Delta_{min} );
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0, & \mbox{if\ }
 *  \frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0 \\
 *  \Delta_i^{(t-1)}, & \mbox{otherwise}
 *  \end{array}
 *  \f$
 *
 *  The parameters \f$\eta^+ > 1\f$ and \f$0 < \eta^- < 1\f$ control
 *  the speed of the adaptation. To stabilize the increments, they are
 *  restricted to the interval \f$[\Delta_{min}, \Delta_{max}]\f$. <br>
 *  After the adaptation of the \f$\Delta_i\f$ the update for the
 *  weights will be calculated as
 *
 *  \f$
 *  \Delta w_i^{(t)} := - \mbox{sign}
 *  \left( \frac{\partial E^{(t)}}{\partial w_i}\right) \cdot \Delta_i^{(t)}
 *  \f$
 *
 *  For further information about the algorithm, please refer to: <br>
 *
 *  Christian Igel and Michael H&uuml;sken, <br>
 *  "Empirical Evaluation of the Improved Rprop Learning Algorithm". <br>
 *  In Neurocomputing Journal, 2002, in press <br>
 *
 *
 *  \author  C. Igel, M. H&uuml;sken
 *  \date    1999
 *
 *  \par Changes:
 *      none
 *
 *  \par Status:
 *      stable
 *
 */
class IRpropMinus : public RpropMinus {
public:
	IRpropMinus();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "IRpropMinus"; }

	void step(const ObjectiveFunctionType& objectiveFunction);
};

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef IRpropPlus Rprop99;

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef IRpropMinus Rprop99d;

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef RpropPlus Rprop93;

//! Used to connect the class names with the year of
//! publication of the paper in which the algorithm was introduced.
typedef RpropMinus Rprop94;
}

#endif

