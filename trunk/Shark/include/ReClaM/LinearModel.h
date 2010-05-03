//===========================================================================
/*!
*  \file LinearModel.h
*
*  \brief Linear models on a real vector space
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
*  The #LinearFunction class provides a simple model, that is,
*  a linear function on a real vector space (a map to the reals).
*  The #LinearMap class provides a linear map from one real
*  vector space to another.
*
*
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

#ifndef _LinearModel_H_
#define _LinearModel_H_


#include <Array/Array2D.h>
#include <ReClaM/Model.h>


//!
//! The LinearMap class represents a simple linear model
//! \f$ f : R^n \rightarrow R^m, x \mapsto A x \f$
//! where the matrix A makes up the parameters of the model.
//!
class LinearMap : public Model
{
public:
	LinearMap(int inputdim, int outputdim);
	~LinearMap();

	void model(const Array<double>& input, Array<double>& output);
	void modelDerivative(const Array<double>& input, Array<double>& derivative);
};


//!
//! The LinearMap class represents a simple affine linear model
//! \f$ f : R^n \rightarrow R^m, x \mapsto A x + b \f$
//! where the matrix A and the vector b make up the parameters
//! of the model.
//!
class AffineLinearMap : public Model
{
public:
	AffineLinearMap(int inputdim, int outputdim);
	~AffineLinearMap();

	void model(const Array<double>& input, Array<double>& output);
	void modelDerivative(const Array<double>& input, Array<double>& derivative);
};


//!
//! The LinearFunction class represents a simple linear function
//! \f$ f : R^n \rightarrow R, x \mapsto \langle v, x \rangle \f$
//! where the vector v makes up the parameters of the model.
//!
class LinearFunction : public LinearMap
{
public:
	LinearFunction(int dimension);
};


//!
//! The LinearFunction class represents a simple linear function
//! \f$ f : R^n \rightarrow R, x \mapsto \langle v, x \rangle + b \f$
//! where the vector v and the offset b make up the parameters
//! of the model.
//!
class AffineLinearFunction : public AffineLinearMap
{
public:
	AffineLinearFunction(int dimension);
};


//!
//! The LinearClassifier class is a multi class classifier model
//! suited for linear discriminant analysis. For c classes
//! \f$ 0, \dots, c-1 \f$ the model holds class mean vectors
//! \f$ m_c \f$ and a shared data scatter matrix \f$ M \f$. It
//! predicts the class of a vector x according to the rule
//! \f$ \textrm{argmin}_{c} (x - m_c)^T M^{-1} (x - m_c) \f$.
//! The output is a unit vector in \f$ R^c \f$ composed of zeros
//! and a single 1 entry in the component corresponding to the
//! predicted class.
//!
class LinearClassifier : public Model
{
public:
	LinearClassifier(int dimension, int classes);
	~LinearClassifier();

	void setParameter(unsigned int index, double value);
	void model(const Array<double>& input, Array<double>& output);

	inline int getNumberOfClasses() const
	{
		return numberOfClasses;
	}

protected:
	int numberOfClasses;
	Array<double> mean;
	Array2D<double> covariance;
	Array2D<double> inverse;
	bool bNeedsUpdate;
};


#endif

