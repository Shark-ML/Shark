//===========================================================================
/*!
 *  \file ArtificialDistributions.h
 *
 *  \brief Artificial benchmark data
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
//===========================================================================


#ifndef _ArtificialDistributions_H_
#define _ArtificialDistributions_H_


#include <ReClaM/Dataset.h>
#include <Array/Array2D.h>


//!
//! \brief Distribution of the chessboard classification problem
//!
//! \par
//! The n-dimensional chessboard problem of size d
//! is defined as the uniform distribution on the
//! n-dimensional cube \f$ x \in [0, d[^n \f$ with
//! associated label
//! \f$ y = (-1)^{\sum_{i=1}^n \lfloor x_i \rfloor} \f$.
//!
//! \par
//! The Bayes optimal classifier for this distribution
//! makes zero error, that is, the distribution is not
//! noisy. It is nevertheless difficult, as the
//! labeling rule splits the input space into
//! \f$ d^n \f$ connected components.
//!
class Chessboard : public DataSource
{
public:
	//! Constructor
	Chessboard(int dim = 2, int size = 4);

	//! Destructor
	~Chessboard();


	//! This function generates examples drawn
	//! i.i.d. from the chessboard distribution.
	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	//! chess board size
	int size;
};


//!
//! \brief Noisy version of the chessboard classification problem
//!
//! \par
//! The n-dimensional noisy chessboard problem of size d
//! is defined as a sum of radial Gaussian distributions
//! centered on a \f$ n \times \dots \times n \f$ grid
//! in \f$ R^d \f$. All Gaussians have equal weights
//! \f$ 1/n^d \f$ and neighboring grid points correspond
//! to opposite class labels.
//!
class NoisyChessboard : public Chessboard
{
public:
	//! Constructor
	NoisyChessboard(int dim = 2, int size = 4, double noiselevel = 0.4);

	//! Destructor
	~NoisyChessboard();


	//! This function generates examples drawn
	//! i.i.d. from the chessboard distribution.
	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	//! standard deviation of the Gaussians
	double noiselevel;
};


//!
//! \brief Distribution of the noisy interval problem
//!
class NoisyInterval : public DataSource
{
public:
	//! Constructor
	NoisyInterval(double bayesRate, int dimensions = 1);

	//! Destructor
	~NoisyInterval();


	//! This function generates examples drawn
	//! i.i.d. from the distribution.
	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	double bayesRate;
	int dimensions;
};


//!
//! \brief Spherical Distribution
//!
//! Distribution in R^n with spherical symmetry
//! around the origin. The radius component is
//! uniformly distributed in the set \f$[0, 1] \cup [2, 3]\f$
//! and the components of the space make up the
//! classes.
//! This problem is usually simple as the classes
//! are clearly separated.
class SphereDistribution1 : public DataSource
{
public:
	//! Constructor
	SphereDistribution1(int dim);

	//! Destructor
	~SphereDistribution1();


	//! This function generates examples drawn
	//! i.i.d. from the distribution.
	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	int dimension;
};


//!
//! \brief Sparse Distribution
//!
//! (2n+m)-dimensional distribution with sparse binary values.
//! Every example from this distribution consists of a vector
//! with k ones and 2n+m-k zeros. The first one is within the
//! first n coordinates for positive and in the second n variables
//! for negative class examples. The remaining ones are drawn
//! independently among the remaining variables.
//!
class SparseDistribution : public DataSource
{
public:
	//! Constructor
	SparseDistribution(int n, int m, int k);

	//! Destructor
	~SparseDistribution();


	//! This function generates examples drawn
	//! i.i.d. from the distribution.
	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	int dim1;
	int dim2;
	int num;
};


//!
//! \brief linear transformation of data
//!
//! The TransformedProblem class applies a linear transformation
//! to the input-part x of the examples (x, y) created by another
//! DataSource object.
class TransformedProblem : public DataSource
{
public:
	//! Constructor
	TransformedProblem(DataSource& source, Array2D<double>& transformation);

	//! Descructor
	~TransformedProblem();


	//! This function generates examples drawn
	//! i.i.d. from the distribution.
	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	DataSource& base;
	Array2D<double> transformation;
};

//! \brief Class modeling a multi-class test problem.
class MultiClassTestProblem : public DataSource
{
public:
	MultiClassTestProblem(double epsilon = 0.5);
	~MultiClassTestProblem();

	bool GetData(Array<double>& data, Array<double>& target, int count);

protected:
	double m_epsilon;
};


#endif

