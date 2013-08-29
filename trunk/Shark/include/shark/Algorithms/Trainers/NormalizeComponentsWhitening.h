//===========================================================================
/*!
 *  \file NormalizeComponentsWhitening.h
 *
 *  \brief Data normalization to zero mean, unit variance and zero covariance 
 *
 *
 *  \author T. Glasmachers
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


#ifndef SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSWHITENING_H
#define SHARK_ALGORITHMS_TRAINERS_NORMALIZECOMPONENTSWHITENING_H


#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>
#include <shark/Data/Statistics.h>
#include <shark/LinAlg/Inverse.h>

namespace shark {


/// \brief Train a linear model to whiten the data
template <class VectorType = RealVector>
class NormalizeComponentsWhitening : public AbstractUnsupervisedTrainer<LinearModel<VectorType, VectorType> >
{
	typedef AbstractUnsupervisedTrainer<LinearModel<VectorType, VectorType> > base_type;
public:
	
	double m_targetVariance;
	NormalizeComponentsWhitening(double targetVariance = 1.0){ 
		m_targetVariance = targetVariance;
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "NormalizeComponentsWhitening"; }

	void train(LinearModel<VectorType, VectorType>& model, UnlabeledData<VectorType> const& input){
		SHARK_CHECK(input.numberOfElements() >= 2, "[NormalizeComponentsWhitening::train] input needs to contain at least two points");
		std::size_t dc = dataDimension(input);
		model.setStructure(dc, dc, true, false); // dense model with bias having input and output dimension equal to data dimension

		VectorType mean;
		RealMatrix covariance;
		meanvar(input, mean, covariance);
		
		//we use the inversed cholesky decomposition for whitening
		//since we have to assume that covariance does not have full rank, we use
		//the generalized decomposition
		RealMatrix UInv;
		decomposedGeneralInverse(covariance,UInv);
		UInv*=std::sqrt(m_targetVariance);
		//we need the transpose of UInv. for that we reuse the memory of covariance
		noalias(covariance)=trans(UInv);
		
		
		VectorType offset(dc);
		fast_prod(covariance,mean,offset,false, -1.0);
		
		model.setStructure(covariance, offset);
	}
};


}
#endif
