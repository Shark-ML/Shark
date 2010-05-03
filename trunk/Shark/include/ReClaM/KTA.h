/*!
*  \file KTA.h
*
*  \brief Implementation of the (negative) Kernel Target Alignment (KTA) 
* 		   as proposed by Nello Cristianini
*
*  \author  T. Glasmachers
*  \date    2006, 2009
*
*  \par
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-25558<BR>
*      Fax:   +49-234-32-14209<BR>
*      eMail: shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
*      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
*      <BR> 
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


#ifndef _KTA_H_
#define _KTA_H_


#include <SharkDefs.h>
#include <ReClaM/ErrorFunction.h>
#include <ReClaM/KernelFunction.h>
#include <ReClaM/Svm.h>


/*!
 *  \brief Implementation of the negative Kernel Target
 *  Alignment (KTA) as proposed by Nello Cristianini.
 *  This measure is extended for multi-class problems.
 *
 *  \par
 *  Kernel Target Alignment measures how well a kernel
 *  fits a classification training set. It is invariant
 *  under kernel rescaling. To turn it into an error
 *  function (i.e., to make minimization meaningful)
 *  the negative KTA is implemented.
 *
 *  \par
 *  The KTA has two important properties: It is
 *  differentiable and independent of the actual
 *  classifier.
 *
 *  \par
 *  The KTA as originally proposed by Nello Cristianini
 *  is not properly arranged for unbalanced datasets.
 *  Thus, the NegativeBKTA (balanced kernel target
 *  alignment) class implements an invariant version,
 *  which is preferable for unbalanced datasets.
 *
 *  \par
 *  KTA measures the similarity, in terms of the inner
 *  product, of the kernel Gram matrix K with a perfect
 *  Gram matrix D with entries 1 or -1 for examples of
 *  coinciding or different label, respectively. Then
 *  the kernel target alignment is given by
 *  \f[
 *  	\hat A = \frac{\langle D, K \rangle}{\sqrt{\langle D, D \rangle \langle K, K\rangle}}
 *  \f]
 *  We generalize the measure by using the value -1/(N-1)
 *  for entries corresponding to different classes, which
 *  gives a canonical and symmetric generalization. Here,
 *  N denotes the number of classes.
 */
class NegativeKTA : public ErrorFunction
{
public:
	//! Constructor
	NegativeKTA(unsigned int numberOfClasses = 2);

	//! Destructor
	~NegativeKTA();


	//! Computes the negative Kernel Target Alignment between
	//! the target and the kernel function output on the input.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computes the negative Kernel Target Alignment between
	//! the target and the kernel function output on the input.
	//! The partial derivatives of the negative KTA w.r.t. the
	//! model parameters are returned in the derivative parameter.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);

protected:
	unsigned int m_numberOfClasses;
	double m_offdiag;
	double m_offdiag2;
	double m_2offdiag;
	double m_2offdiag2;
};


//!
//! \brief Balanced version of the #NegativeKTA.
//!
//! \par The Balanced Kernel Target Alignment measure is a variant
//! of the Kernel Target Alignment.
//! This version of the measure is invariant under the fractions of
//! positive and negative examples, and is currently only valid for
//! binary classification problems.
//!
//! \par It is computed as follows:
//! Let p be the number of positive examples and let n be the number
//! of negative examples. We define an inner product between kernel
//! (Gram) matrices M and N as
//! \f[
//!     \langle M, N \rangle := \sum_{i, j} \lambda_{ij} M_{ij} N_{ij}
//! \f]
//! with positive coeffitients
//! \f$ \lambda_{ij} = n/p \f$ if \f$ y_i = y_j = +1 \f$,
//! \f$ \lambda_{ij} = 1 \f$ if \f$ y_i \not= y_j \f$ and
//! \f$ \lambda_{ij} = p/n \f$ if \f$ y_i = y_j = -1 \f$.
//!
//! Then we apply the usual definition of the kernel target alignment
//! \f[
//!     \hat A = \frac{\langle yy^T, K \rangle}{\|yy^T\| \cdot \|K\|}
//! \f]
//! where the norm \f$ \| M \| = \sqrt{\langle M, M \rangle} \f$ is
//! defined according to the inner product defined above.
//!
//! \par To my knowledge, the balancing modification has not been
//! proposed in the literature so far.
//!
//! \author: T. Glasmachers
//!
//! \date: 2006
//!
class NegativeBKTA : public ErrorFunction
{
public:
	//! Constructor
	NegativeBKTA();

	//! Destructor
	~NegativeBKTA();


	//! Computes the negative Balanced Kernel Target Alignment between
	//! the target and the kernel function output on the input.
	double error(Model& model, const Array<double>& input, const Array<double>& target);

	//! Computes the negative Balanced Kernel Target Alignment between
	//! the target and the kernel function output on the input.
	//! The partial derivatives of the negative BKTA w.r.t. the
	//! model parameters are returned in the derivative parameter.
	double errorDerivative(Model& model, const Array<double>& input, const Array<double>& target, Array<double>& derivative);
};


#endif

