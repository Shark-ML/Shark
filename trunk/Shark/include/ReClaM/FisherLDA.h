//===========================================================================
/*!
 *  \file FisherLDA.h
 *
 *  \brief FisherLDA
 *
 *  \par
 *      This file offers \em Fisher's \em Linear \em Discriminant \em Analysis
 *      which is used to compress data for a better visualization and analysis.
 *
 *  \author B. Weghenkel
 *  \date 2007
 *
 *  \par Copyright (c) 1998-2009:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
 *
 *  \par Project:
 *      ReClaM
 *
 *
 *  <BR>
 *
 *
 *  <BR><HR>
 *  This file is part of ReClaM. This library is free software;
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


#ifndef _FISHERLDA_H_
#define _FISHERLDA_H_


#include <LinAlg/VecMat.h>
#include <ReClaM/LinearModel.h>
#include <ReClaM/Optimizer.h>


//===========================================================================
/*!
 * \brief Fisher's Linear Discriminant Analysis for data compression
 *
 * Similar to PCA, \em Fisher's \em Linear \em Discriminant \em Analysis is a
 * method for reducing the datas dimensionality. In contrast to PCA it also uses
 * class information.
 *
 * Consider the datas covariance matrix \f$ S \f$ and a unit vector \f$ u \f$ 
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
 * \b Note: the max. dimensionality for the subspace is \#NumOfClasses-1.
 *
 * For more detailed information about Fisher-LDA, see \e Bishop, \e Pattern
 * \e Recognition \e and \e Machine \e Learning.
 */
class FisherLDA : public Optimizer
{
public:
	FisherLDA();
	~FisherLDA();

	void init(Model& model);
	void init(bool whitening=false);

	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target);
	double optimize(AffineLinearMap& model, const Array<double>& input, const Array<double>& target);
	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target, Array<double> &eigenvalues, Array<double> &eigenvectors);
	double optimize(AffineLinearMap& model, const Array<double>& input, const Array<double>& target, Array<double> &eigenvalues, Array<double> &eigenvectors);

protected:
	void MeanAndScatter(AffineLinearMap& model, const Array<double>& input, const Array<double>& target, Vector& mean, Matrix& scatter);

	bool bWhitening;
};


#endif

