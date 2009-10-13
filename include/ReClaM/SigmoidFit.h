//===========================================================================
/*!
 *  \file SigmoidFit.h
 *
 *  \brief Optimization of the #SigmoidModel according to Platt, 1999
 *
 *  \author  T. Glasmachers
 *  \date    2006
 *
 *  \par Copyright (c) 1999-2006:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
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


#ifndef _SigmoidFit_H_
#define _SigmoidFit_H_


#include <SharkDefs.h>
#include <ReClaM/Optimizer.h>


//!
//! \brief Optimize a sigmoid after SVM outputs to turn them
//!        into probability estimates.
//!
//! \par
//! The SigmoidFit class implements a non-iterative optimizer.
//! It optimizes a sigmoid to fit the posterior probability for
//! positive class given SVM outputs. Although the optimizer is
//! iterative in its nature, it is more convenient to implement
//! it in one single step. As a side effect, Platt's original and
//! very heuristic stopping conditions are kept.
//!
//! \par
//! The method was presented by John Platt in 1999, see<br>
//! <i>Probabilistic Outputs for Support Vector Machines
//! and Comparisons to Regularized Likelyhood Methods,
//! Advances in Large Margin Classifiers, pp. 61-74,
//! MIT Press, (1999).</i><br>
//! The full paper can be downloaded from<br>
//! <i>http://www.research.microsoft.com/~jplatt</i><br>
//! --- pseudo-code is given in the paper.
//!
class SigmoidFit
{
public:
	//! Constructor
	SigmoidFit();

	//! Destructor
	~SigmoidFit();


	//! The init method does nothing usefull
	//! but fulfills the base class interface.
	void init(Model& model);

	//! Don't call the optimize member more than once,
	//! as the SigmoidFit optimizer is not iterative.
	//! \param  model          reference to the #SigmoidModel object to train
	//! \param  errorfunction  unused - can be set to #SVM_Optimizer::dummyError.
	//! \param  input          #SVM output, real-valued
	//! \param  target         binary labels +1 or -1
	double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);

	//! Compute the maximum of -200 and the logarithm of x.
	double mylog(double x);
};


#endif

