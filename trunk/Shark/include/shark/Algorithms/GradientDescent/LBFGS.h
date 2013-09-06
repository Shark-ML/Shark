//===========================================================================
/*!
 *  \brief LBFGS
 *
 *  The Limited-Memory Broyden, Fletcher, Goldfarb, Shannon (BFGS) algorithm
 *  is a quasi-Newton method for unconstrained real-valued optimization.
 *  See: http://en.wikipedia.org/wiki/LBFGS for details.
 *
 *  \author S. Dahlgaard, O.Krause
 *  \date 2013
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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


#ifndef SHARK_ML_OPTIMIZER_LBFGS_H
#define SHARK_ML_OPTIMIZER_LBFGS_H

#include <shark/Algorithms/GradientDescent/AbstractLineSearchOptimizer.h>

#include <deque>

namespace shark {

//! \brief Limited-Memory Broyden, Fletcher, Goldfarb, Shannon algorithm for unconstrained optimization
class LBFGS : public AbstractLineSearchOptimizer{
protected:
	void initModel();
	void computeSearchDirection();
public:
	LBFGS() :m_numHist(100){}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "LBFGS"; }
	
	///  \brief Specify the amount of steps to be memorized and used to find the L-BFGS direction.
	///
	///\param numhist The amount of steps to use.
	void setHistCount(unsigned int numhist) {
		SHARK_CHECK(numhist > 0, "[LBFGS::setHistCount] An empty history is not allowed");
		m_numHist = numhist;
	}

	//from ISerializable
	void read(InArchive &archive);
	void write(OutArchive &archive) const;
protected: // Methods

	///\brief Stores another step and searchDirection, discarding the oldest on if necessary.
	///
	/// \param step Last performed step
	/// \param y difference in gradients
	void updateHist(RealVector& y, RealVector &step);
	/// \brief Get the LBFGS direction. 
	///
	/// This approximates the inverse hessian multiplied by the gradient.
	/// This uses the rho, alpha and beta vectors. Description of these
	/// can be seen in ie. the wiki page of LBFGS.
	void getDirection(RealVector& searchDirection);


protected: // Instance vars
	double m_updThres;///<Threshold for when to update history.
	unsigned int m_numHist; ///< Number of steps to use for LBFGS.
	// Initial Hessian approximation. We use a diagonal matrix, where each element is
	// the same, so we only need to store one double.
	double          m_hdiag;

	// Saved steps for creating the approximation.
	// Use deque as it gives fast pop.front, push.back and access. Supposedly.
	// steps holds the values x_(k+1) - x_k
	// gradientDifferences holds the values g_(k+1) - g_k
	std::deque<RealVector> m_steps;
	std::deque<RealVector> m_gradientDifferences;
};

}
#endif
