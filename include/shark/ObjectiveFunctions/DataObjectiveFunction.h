//===========================================================================
/*!
 *  \brief data-dependent objective functions for learning
 *
 *
 *  \author O. Krause
 *  \date 2010-2011
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

#ifndef SHARK_OBJECTIVEFUNCTIONS_DATAOBJECTIVEFUNCTION_H
#define SHARK_OBJECTIVEFUNCTIONS_DATAOBJECTIVEFUNCTION_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Core/SearchSpaces/VectorSpace.h>
#include <shark/Data/Dataset.h>

namespace shark {


/// \brief Data-dependent objective function for supervised learning.
///
/// \par
/// The SupervisedObjectiveFunction class is a general super
/// class of all objective functions that depend on data for
/// supervised learning. Such functions are omnipresent in
/// machine learning in the form of "error" or "empirical risk"
/// terms.
template<class InputType, class LabelType>
class SupervisedObjectiveFunction : public AbstractObjectiveFunction< VectorSpace<double>, double >
{
public:
	typedef LabeledData<InputType, LabelType> DatasetType;
	SupervisedObjectiveFunction()
	{ }

	virtual ~SupervisedObjectiveFunction()
	{ }

	virtual void setDataset(DatasetType const& dataset) = 0;
};


/// \brief Data-dependent objective function for unsupervised learning.
///
/// \par
/// The UnsupervisedObjectiveFunction class is a general super
/// class of all objective functions that depend on data for
/// unsupervised learning.
template<class InputType>
class UnsupervisedObjectiveFunction : public AbstractObjectiveFunction< VectorSpace<double>, double >
{
public:
	typedef UnlabeledData<InputType> DatasetType;
	virtual void setData(DatasetType const & dataset) = 0;
};


}
#endif
