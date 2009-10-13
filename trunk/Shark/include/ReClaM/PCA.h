//===========================================================================
/*!
 *  \file PCA.h
 *
 *  \brief Principal Component Analysis (PCA)
 *
 *  \par
 *      This file offers \em Principal \em Component \em Analysis
 *      which is used to compress data for a better visualization
 *      and analysis.
 *
 *  \par
 *      This implementation is based on a class removed from the
 *      LinAlg package, written by M. Kreutz in 1998.
 *
 *  \author T. Glasmachers
 *  \date 2007
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


#ifndef _PCA_H_
#define _PCA_H_


#include <ReClaM/LinearModel.h>
#include <ReClaM/Optimizer.h>


//===========================================================================
/*!
 *  \brief "Principal Component Analysis" class for data compression.
 *
 *  The \em Principal \em Component \em Analysis also known as
 *  \em Karhunen-Loeve-Transformation takes a symmetric
 *  \f$ n \times n \f$ matrix \f$ A \f$ and uses its decomposition
 *
 *  \f$
 *      A = \Gamma \Lambda \Gamma^T
 *  \f$
 *
 *  where \f$ \Lambda \f$ is the diagonal matrix of eigenvalues
 *  of \f$ A \f$ and \f$ \Gamma \f$ is the orthogonal matrix
 *  with the corresponding eigenvectors as columns.
 *  \f$ \Lambda \f$ then defines a successive orthogonal rotation,
 *  that maximizes
 *  the variances of the coordinates, i.e. the coordinate system
 *  is rotated in such a way that the correlation between the new
 *  axes becomes zero. If there are \f$ p \f$ axes, the first
 *  axis is rotated in a way that the points on the new axis
 *  have maximum variance. Then the remaining \f$ p - 1 \f$
 *  axes are rotated such that a another axis covers a maximum
 *  part of the rest variance, that is not covered by the first
 *  axis. After the rotation of \f$ p - 1 \f$ axes,
 *  the rotation destination of axis no. \f$ p \f$ is fixed.
 *  An application for \em PCA is the reduction of dimensions
 *  by skipping the components with the least corresponding
 *  eigenvalues/variances.
 *
 */
class PCA : public Optimizer
{
public:
	PCA();
	~PCA();

	void init(Model& model);
	void init(bool whitening);

	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target);
	double optimize(AffineLinearMap& model, const Array<double>& input);
	double optimize(Model& model, ErrorFunction& error, const Array<double>& input, const Array<double>& target, Array<double> &eigenvalues, Array<double> &eigenvectors);
	double optimize(AffineLinearMap& model, const Array<double>& input, Array<double> &eigenvalues, Array<double> &eigenvectors);

protected:
	void MeanAndScatter(const Array<double>& input, Array<double>& mean, Array2D<double>& scatter);

	bool bWhitening;
};


#endif

