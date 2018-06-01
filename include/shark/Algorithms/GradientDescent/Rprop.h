/*!
 * 
 *
 * \brief       implements different versions of Resilient Backpropagation of error.
 * 
 * \author      Oswin Krause
 * \date        2010
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

#ifndef SHARK_ML_OPTIMIZER_RPROP_H
#define SHARK_ML_OPTIMIZER_RPROP_H

#include <shark/Algorithms/AbstractSingleObjectiveOptimizer.h>

namespace shark{

/*!
 *  \brief This class offers methods for the usage of the
 *         Resilient-Backpropagation-algorithm with/out weight-backtracking.
 *
 *  The Rprop algorithm is an improvement of the algorithms with adaptive
 *  learning rates, which use increments for the update
 *  of the weights which are independent from the absolute partial
 *  derivatives. This makes sense, because large flat regions
 *  in the search space (plateaus) lead to small absolute partial
 *  derivatives and so the increments are chosen small, but the increments
 *  should be large to skip the plateau. In contrast, the absolute partial
 *  derivatives are very large at the "slopes" of very "narrow canyons",
 *  which leads to large increments that will skip the minimum lying
 *  at the bottom of the canyon, but it would make more sense to
 *  chose small increments to hit the minimum.
 *
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
 * There are several variants of the algorithm depending on what happens
 * when the optimum is overstepped, i.e. a sign change of the gradient occurs
 * and/or the new objective value is larger than the old.
 *
 *  Weight-backtracking can be used to increase the
 *  stability of the method.
 *  if \f$\frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0\f$ then
 *  \f$\Delta w_i^{(t)} := - \Delta w_i^{(t-1)};
 *  This heuristic can be improved by further taking the value of the last iteration
 *  into ccount: only undo an updated if the sign changed and the new function value 
 *  is worse than the last. The idea of this modification is, that a change of the sign of the
 *  partial derivation \f$\frac{\partial E}{\partial w_i}\f$
 *  only states, that a minimum was skipped and not, whether this step
 *  lead to an approach to the minimum or not.
 *
 *  Furthermore, it has been shown to be beneficial to use gradient freezing
 *  when the rgadient changes sign, i.e. ,
 *  if \f$\frac{\partial E^{(t-1)}}{\partial w_i} \cdot
 *  \frac{\partial E^{(t)}}{\partial w_i} < 0\f$ then
 *  \frac{\partial E^{(t)}}{\partial w_i} := 0\f$;
 * Thus, after an unsuccessful step is performed, delta is not changed
 * for one iteration.
 *
 * Based on this, 4 major algorithms can be derived:
 * Rprop-: (no backtracking, no freezing)
 * IRprop-: (no backtracking, freezing)
 * Rprop+: (gradient based backtracking, freezing)
 * IRprop+: (function value based backtracking, freezing)
 *
 * By default, IRprop+ is chosen.
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
 *  Martin Riedmiller, <br>
 *  "Advanced Supervised Learning in Multi-layer Perceptrons -
 *  From Backpropagation to Adaptive Learning Algorithms". <br>
 *  In "International Journal of Computer Standards and Interfaces", volume 16,
 *  no. 5, 1994, pp. 265-278 <br>
 *
 *  Christian Igel and Michael H&uuml;sken, <br>
 *  "Empirical Evaluation of the Improved Rprop Learning Algorithm". <br>
 *  In Neurocomputing Journal, 2002, in press <br>
 * \ingroup gradientopt
 */
template<class SearchPointType = RealVector>
class Rprop : public AbstractSingleObjectiveOptimizer<SearchPointType >
{
public:
	typedef AbstractObjectiveFunction<SearchPointType,double> ObjectiveFunctionType;
	Rprop();

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Rprop"; }

	void init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint);
	void init(ObjectiveFunctionType const& objectiveFunction, SearchPointType const& startingPoint, double initDelta);
	using AbstractSingleObjectiveOptimizer<SearchPointType >::init;

	void step(ObjectiveFunctionType const& objectiveFunction);

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
	
	void setUseOldValue(bool useOldValue){
		m_useOldValue = useOldValue;
		if(m_useOldValue)
			this->m_features |= this->REQUIRES_VALUE;
		else
			this->m_features.reset(this->REQUIRES_VALUE);
	}
	void setUseFreezing(bool useFreezing){
		m_useFreezing = useFreezing;
	}
	void setUseBacktracking(bool useBacktracking){
		m_useBacktracking = useBacktracking;
	}

	//! return the maximal step size component
	double maxDelta() const {
		return *std::max_element(m_delta.begin(),m_delta.end());
	}
	
	/// \brief Returns the derivative at the current point. Can be used for stopping criteria.
	SearchPointType const& derivative()const{
		return m_derivative;
	}
protected:
	SearchPointType m_derivative;

	//! The increase factor \f$ \eta^+ \f$, set to 1.2 by default.
	double m_increaseFactor;

	//! The decrease factor \f$ \eta^- \f$, set to 0.5 by default.
	double m_decreaseFactor;

	//! The upper limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 1e100 by default.
	double m_maxDelta;

	//! The lower limit of the increments \f$ \Delta w_i^{(t)} \f$, set to 0.0 by default.
	double m_minDelta;
	//! The last function value observed
	double m_oldValue;

	size_t m_parameterSize;

	//! The last error gradient.
	SearchPointType m_oldDerivative;
	//! the step eprformed last. used for weight backtracking
	SearchPointType m_deltaw;

	//! The absolute update values (increment) for all weights.
	SearchPointType m_delta;
	
	bool m_useFreezing;
	bool m_useBacktracking;
	bool m_useOldValue;
};

extern template class Rprop<RealVector>;
extern template class Rprop<FloatVector>;

}

#endif

