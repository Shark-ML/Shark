//===========================================================================
/*!
 *  \file NetParams.h
 *
 *  \brief Easily configuration file reading for neural networks
 *
 *  \author  C. Igel
 *  \date    1999
 *
 *  \par Copyright (c) 1999-2000
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
 *
 *
 */
//===========================================================================

#ifndef __NETPARAMS_H
#define __NETPARAMS_H


#include <FileUtil/Params.h>
#include <ReClaM/BFGS.h>


/*!
 *  \brief Offers functions for easily reading information about
 *         a network, an error measure and an optimization algorithm
 *         from a configuration file.
 *
 *  A structure of predefined variables can be used to store information read
 *  from a configuration file. These information can be used to create
 *  a network, define an error measure and initialize an optimization
 *  algorithm, to automize the usage of the ReClaM library for a personal
 *  network.
 */
class NetParams : public Params
{
public:

	//! Reads the values for all parameters defined in #io
	//! from the configuration file given in "argv".
	NetParams(int argc = 0, char **argv = NULL);

	//! Name of the file containing the structure information
	//! for the network used.
	std::string netFilename;

	//! Name of the file containing the input patterns and
	//! their corresponding target values used for the
	//! training of the network.
	std::string trainDataFilename;

	//! Name of the file containing the input patterns and
	//! their corresponding target values used for
	//! testing the network.
	std::string testDataFilename;

	//! Name of the file containing the input patterns and
	//! their corresponding target values used for
	//! validation of the trained network.
	std::string validateDataFilename;

	//! A prefix, that is added to the name of output files.
	std::string prefix;

	//!
	std::string function;

	//! If set to "true", the Rprop algorithm is used for
	//! optimization of the network (see Rprop.h).
	bool        useRprop;

	//! Will the data be normalized?
	bool        norm;

	//! Will the data be normalized by variance?
	bool        normByVariance;

	//! Will an initialization take place?
	bool        init;

	//! If set to "true", the network uses linear activation functions
	//! for the output neurons.
	bool        linearOutput;

	//! Will the result be plotted?
	bool        plot;

	//! The no. of time steps, after which the current results will
	//! be displayed.
	unsigned    interval;

	//! The no. of generations that will be trained.
	unsigned    cycles;

	//! The no. of runs that will take place.
	unsigned    runs;

	//! The learning rate \f$\eta\f$ used by SteepestDescent
	//! and StochasticGradientDescent.
	double      lr;

	//! Controls the influence of the momentum term when updating
	//! the weight values.
	double      momentum;

	//!
	double weightDecay;

	//! Initialization value for the random number generator.
	int         seed;

	//! Increase factor of the stepsize \f$|\Delta w_i|\f$ for
	//! the Rprop algorithm (see Rprop.h).
	double      np;

	//! Decrease factor of the stepsize \f$|\Delta w_i|\f$ for
	//! the Rprop algorithm (see Rprop.h).
	double      nm;

	//! Initial value for the stepsize \f$|\Delta w_i|\f$
	//! for the Rprop algorithm (see Rprop.h).
	double      delta0;

	//! Lower limit for the stepsizes \f$|\Delta w_i|\f$ for
	//! the Rprop algorithm (see Rprop.h).
	double      dMin;

	//! Upper limit for the stepsizes \f$|\Delta w_i|\f$ for
	//! the Rprop algorithm (see Rprop.h).
	double      dMax;

	//! Minimum value, when initializing the weights of a network.
	double      low;

	//! Maximum value, when initializing the weights of a network.
	double      high;

	//! The type of line search algorithm used by the BFGS and
	//! the conjugate gradient optimization algorithms.
	//! "0" = dlinmin, "1" = linmin, "2" = cubic line search;
	unsigned    lineSearch;

	//! The value of the "left" bracket used for line search,
	//! with \f$ax < bx\f$ (see BFGS).
	double      ax;

	//! The value of the "right" bracket used for line search,
	//! with \f$bx > ax\f$ (see BFGS).
	double      bx;

	//! Controls the accuracy of the line search (see BFGS).
	double      lambda;

	//! Defines FileUtil::io_strict to be used, when dealing
	//! with the configuration file.
	void set_format_strict();

	//! Defines FileUtil::io to be used, when dealing
	//! with the configuration file.
	void set_format_liberate();


private:


	//! This method is called by the constructor and defines
	//! the default values token names and variables for
	//! all parameters.
	void io(std::istream&, std::ostream&, FileUtil::iotype);

	//! Used to define the type of io-function from FileUtil
	//! that is used. "true" means, that FileUtil::io_strict
	//! is used, "false" will cause the usage of FileUtil::io
	//! (this is also the default value set by the constructor).
	bool strict;

};

#endif /* !__NETPARAMS_H */


