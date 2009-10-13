//===========================================================================
/*!
 *  \file SteepestDescent.h
 *
 *  \brief The simplest learning strategy.
 *
 *  The steepest descent strategy uses the calculated error derivatives
 *  to adjust the weights of the network.
 *
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2001:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *
 *
 *
 *  <BR><HR>
 *  This file is part of Shark. This library is free software;
 *  you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software
 *  Foundation; either version 2, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, write to the Free Software
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
//===========================================================================

#ifndef STEEPEST_DESCENT_H
#define STEEPEST_DESCENT_H

#include <Array/Array.h>
#include <Array/ArrayOp.h>
#include <ReClaM/Optimizer.h>

//===========================================================================
/*!
 *  \brief Standard steepest descent.
 *
 *  \author  C. Igel
 *  \date    1999
 *
 */
class SteepestDescent : public Optimizer
{
public:
	/*!
	 *  \brief default constructor
	 */
	SteepestDescent() {
		m_lr = 0.;
		m_mu = 0.;
		m_online = false;
	}

	void init(Model& model)
	{
		init(model, 0.1, 0.3, false);
	}

	/*!
	 *      \param  model   ReClaM Model to optimize
	 *      \param  lr      Learning rate \f$ \eta \f$.
	 *      \param  mu      Momentum parameter \f$ \mu \f$.
	 *      \param  online  Learn in online or batch mode
	 */
	void init(Model& model, double lr, double mu, bool online)
	{
		m_path.resize(model.getParameterDimension());
		m_path = 0.0;

		m_lr = lr;
		m_mu = mu;
		m_online = online;
	}

	/*!
	 *  \brief get learning rate 
	 */
	double getLearningRate() const {
		return m_lr;
	}

	/*!
	 *  \brief set learning rate 
	 */
	void setLearningRate(double lr) {
		m_lr = lr;
	}

	/*!
	 *  \brief get momentum parameter 
	 */
	double getMomentum() const {
		return m_mu;
	}

	/*!
	 *  \brief set momentum parameter 
	 */
	void setMomentum(double mu) {
		m_mu = mu;
	}


//===========================================================================
	/*!
	 *  \brief Iterative updating of the weights either
	 *  by presenting all patterns (batch) or by presenting
	 *  the patterns one at a time (online).
	 */
	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target)
	{
		double ret = 0.0;
		Array<double> dedw;
		unsigned int p, pc = model.getParameterDimension();

		dedw.resize(pc, false);
		if (!m_path.ndim()) {
			m_path.resize(dedw);
			m_path = 0;
		}

		if (m_online)
		{
			if (input.ndim() == 1)
			{
				ret = error.errorDerivative(model, input, target, dedw);
				m_path = -m_lr * dedw + m_mu * m_path;
				for (p=0; p<pc; p++) model.setParameter(p, model.getParameter(p) + m_path(p));
			}
			else
			{
				for (unsigned pattern = 0; pattern < input.dim(0); ++pattern)
				{
					ret += error.errorDerivative(model, input[pattern], target[pattern], dedw);
					m_path = -m_lr * dedw + m_mu * m_path;
					for (p=0; p<pc; p++) model.setParameter(p, model.getParameter(p) + m_path(p));
				}
			}
		}
		else
		{
			ret = error.errorDerivative(model, input, target, dedw);
			m_path = -m_lr * dedw + m_mu * m_path;
			for (p=0; p<pc; p++) model.setParameter(p, model.getParameter(p) + m_path(p));
		}

		return ret;
	}

	//===========================================================================
	/*!
	 *  \brief Clears  array of momentum values.
	 *
	 *  \return None.
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	void clearMemory()
	{
		m_path = 0;
	};

private:
	Array<double> m_path;

	double m_lr;
	double m_mu;
	bool   m_online;
};


#endif

