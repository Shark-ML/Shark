//===========================================================================
/*!
 * 
 * \file        FisherLDA.h
 *
 * \brief       FisherLDA
 * 
 * 
 *
 * \author      O. Krause
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
//===========================================================================
#ifndef SHARK_ALGORITHMS_TRAINERS_FISHERLDA_H
#define SHARK_ALGORITHMS_TRAINERS_FISHERLDA_H


#include <shark/Models/LinearModel.h>
#include <shark/Algorithms/Trainers/AbstractTrainer.h>

namespace shark {


/*!
 * \brief Fisher's Linear Discriminant Analysis for data compression
 *
 * Similar to PCA, \em Fisher's \em Linear \em Discriminant \em Analysis is a
 * method for reducing the datas dimensionality. In contrast to PCA it also uses
 * class information.
 *
 * Consider the data's covariance matrix \f$ S \f$ and a unit vector \f$ u \f$
 * which defines a one-dimensional subspace of the data. Then, PCA would
 * maximmize the objective \f$ J(u) = u^T S u \f$, namely the datas variance in
 * the subspace. Fisher-LDA, however, maximizes
 * \f[
 *   J(u) = ( u^T S_W u )^{-1} ( u^T S_B u ),
 * \f]
 * where \f$ S_B \f$ is the covariance matrix of the class-means and \f$ S_W \f$
 * is the average covariance matrix of all classes (in both cases, each class'
 * influence is weighted by it's size). As a result, Fisher-LDA finds a subspace
 * in which the class means are wide-spread while (in average) the variance of
 * each class becomes small. This leads to good lower-dimensional
 * representations of the data in cases where the classes are linearly
 * separable.
 *
 * If a subspace with more than one dimension is requested, the above step is
 * executed consecutively to find the next optimal subspace-dimension
 * orthogonally to the others.
 *
 *
 * \b Note: the max. dimensionality for the subspace is \#NumOfClasses-1.
 *
 * It is possible to choose how many dimnsions are used by setting the appropriate value
 * by calling setSubspaceDImension or in the constructor.
 * Also optionally whitening can be applied.
 * For more detailed information about Fisher-LDA, see \e Bishop, \e Pattern
 * \e Recognition \e and \e Machine \e Learning.
 */
class FisherLDA : public AbstractTrainer<LinearModel<>, unsigned int>
{
public:
	/// Constructor
	FisherLDA(bool whitening = false, std::size_t subspaceDimension = 0);

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "Fisher-LDA"; }

	void setSubspaceDimensions(std::size_t dimensions){
		m_subspaceDimensions = dimensions;
	}
	
	std::size_t subspaceDimensions()const{
		return m_subspaceDimensions;
	}

	/// check whether whitening mode is on
	bool whitening() const{ 
		return m_whitening; 
	}

	/// if active, the model whitenes the inputs
	void setWhitening(bool newWhitening){ 
		m_whitening = newWhitening; 
	}

	/// Compute the FisherLDA solution for a multi-class problem.
	void train(LinearModel<>& model, LabeledData<RealVector, unsigned int> const& dataset);

protected:
	void meanAndScatter(LabeledData<RealVector, unsigned int> const& dataset, RealVector& mean, RealMatrix& scatter);
	bool m_whitening;
	std::size_t m_subspaceDimensions;
};


}
#endif
