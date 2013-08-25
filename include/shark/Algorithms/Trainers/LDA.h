//===========================================================================
/*!
 *  \file LDA.h
 *
 *  \brief LDA
 *
 *  \author O. Krause
 *  \date 2010
 *
 *  \par Copyright (c) 1998-2011:
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
#ifndef SHARK_ALGORITHMS_TRAINERS_LDA_H
#define SHARK_ALGORITHMS_TRAINERS_LDA_H


#include <shark/Core/IParameterizable.h>
#include <shark/Models/LinearClassifier.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark {


//!
//! \brief Linear Discriminant Analysis (LDA)
//!
//! \par This class supports two different versions of the LDA. The first is the binary LDA for
//! 2 Classes. It uses boolean class labels and returns a linear model. The other one is a multiclass
//! LDA, which uses vectors as class labels.
//!
class LDA : public AbstractTrainer<LinearClassifier<>, unsigned int>, public IParameterizable
{
public:
	/// constructor
	LDA(double regularization = 0.0){
		setRegularization(regularization);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Linear Discriminant Analysis (LDA)"; }

	/// return the regularization constant
	double regularization()const{
		return m_regularization;
	}

	/// set the regularization constant. 0 means no regularization.
	void setRegularization(double regularization) {
		RANGE_CHECK(regularization >= 0.0);
		m_regularization = regularization;
	}

	/// inherited from IParameterizable; read the regularization parameter
	RealVector parameterVector() const {
		RealVector param(1);
		param(0) = m_regularization;
		return param;
	}
	/// inherited from IParameterizable; set the regularization parameter
	void setParameterVector(RealVector const& param) {
		SIZE_CHECK(param.size() == 1);
		m_regularization = param(0);
	}
	/// inherited from IParameterizable
	size_t numberOfParameters() const {
		return 1;
	}

	//! Compute the LDA solution for a multi-class problem.
	void train(LinearClassifier<>& model, LabeledData<RealVector, unsigned int> const& dataset);

protected:
	//!The regularization parameter \f$ \lambda \f$ adds
	//! \f$ - \lambda I \f$ to the second moment matrix, where
	//! \f$ I \f$ is the identity matrix
	double m_regularization;
};

}
#endif

