//===========================================================================
/*!
 *  \file NoisyRprop.h
 *
 *  \brief Rprop for noisy function evaluations
 *
 *  \author  T. Glasmachers
 *  \date    2007
 *
 *  \par Copyright (c) 1999-2007:
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
 *  <BR>
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


#ifndef _NoisyRprop_H_
#define _NoisyRprop_H_


#include <SharkDefs.h>
#include <Array/Array.h>
#include <ReClaM/Optimizer.h>


//!
//! \brief Rprop-like algorithm for noisy function evaluations
//!
//! \par
//! The Rprop algorithm (see Rprop.h) is a very robust gradient
//! descent based optimizer for real-valued optimization.
//! However, it can not deal with the presence of noice.
//!
//! \par
//! The NoisyRprop algorithm is a completely novel algorithm
//! which tries to carry over the most important ideas from
//! Rprop to the optimization of noisy problems. It is,
//! of course, slower than the Rprop algorithm, but it can
//! handle noisy problems with noisy gradient information.
//!
class NoisyRprop : public Optimizer
{
public:
	//! Constructor
	NoisyRprop();

	//! Destructor
	~NoisyRprop();


	//! initialization with default values
	void init(Model& model);

	//! user defined initialization
	void initUserDefined(Model& model, double delta0P = 0.01);

	//! user defined initialization with
	//! coordinate wise individual step sizes
	void initUserDefined(Model& model, const Array<double>& delta0P);

	//! optimization step
	double optimize(Model& model, ErrorFunction& errorfunction, const Array<double>& input, const Array<double>& target);

protected:
	//! individual step sizes
	Array<double> delta;

	//! minimal episode length
	int minEpisodeLength;

	//! current episode length
	Array<int> EpisodeLength;

	//! fraction 1 to 100 of the episode length
	Array<int> sample;

	//! next 1/100 of the episode length
	Array<int> sampleIteration;

	//! iteration within the current episode
	Array<int> iteration;

	//! normalized position within the episode
	Array<int> position;

	//! "step to the right" event statistic
	Array<int> toTheRight;

	//! leftmost visited place
	Array<int> leftmost;

	//! rightmost visited place
	Array<int> rightmost;

	//! sum of positive derivatives
	Array<double> leftsum;

	//! sum of negative derivatives
	Array<double> rightsum;

	//! current error evaluation
	Array<int> current;

	//! number of error evaluations per iteration
	Array<int> numberOfAverages;

	//! accumulated gradient for each coordinate
	Array<double> gradient;
};


#endif
