//===========================================================================
/*!
*  \file Model.h
*
*  \brief Base class of all models.
*
*  \author  T. Glasmachers, C. Igel
*  \date    2005
*
*  \par Copyright (c)
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

#ifndef _Model_H_
#define _Model_H_

#include <fstream>
#include <SharkDefs.h>
#include <Array/Array.h>


// forward declarations
class ErrorFunction;
class Optimizer;


/*!
 *  \brief Base class of all models.
 *
 * \par
 * ReClaM provides the three base classes Model, ErrorFunction and
 * Optimizer which make up the ReClaM framework for solving regression
 * and classification tasks. This design overrides the ModelInterface
 * design which is kept for backwards compatibility.
 *
 * \par
 * The Model class encapsulates a data processing trainable model.
 * Its internal state is described by an array of parameters.
 *
 * \par
 * A model can be thought of as a parameterized family of mappings
 * \f[ f_p : \mathcal{R}^n \rightarrow \mathcal{R}^m \f]
 * where n and m are the input and output dimensions, respectively.
 * The important special case m=1 is not treated separatly, that is,
 * the trivial output coordinate of 0 has to be supplied for all
 * array operations.
 *
 * \par
 * Models can be operated in two modes:
 *  <ul>
 *    <li>
 *      The model is presented a single input pattern, that is,
 *      a one dimensional #input array. It produces one output
 *      pattern, that is, a one dimensional #output array.
 *    </li>
 *    <li>
 *      The model is presented a set of input patterns (batch mode),
 *      that is, a two dimensional #input array. The first index
 *      corresponds to the pattern index, such that #input[i] is the
 *      i-th input pattern. The model's answer is a two dimensional
 *      output array where again the first index corresponds the
 *      pattern index, that is, #output[i] is the answer of the model
 *      to the input pattern #input[i].
 *    </li>
 *  </ul>
 *
 * \par
 * For gradient based optimization the derivative of the model output
 * with respect to the parameters has to be computed. For the derivative,
 * two rules apply: First, the derivative can only be computed for a
 * single input pattern, that is, not for a batch of patterns. Second,
 * the derivative has the same dimension as the output plus one dimension
 * for the parameters. The parameter dimension always comes last.
 * Thus, the #derivative array is two dimensional. The first
 * dimension represents the model output dimension. The second
 * dimension is indexed by the model parameters.
 * If the derivative of a whole batch of input patterns is needed
 * (as it is usually the case for error derivative computations),
 * the derivative has to be computes sequentially for each input
 * pattern at a time.
 *
 * \par
 * When deriving a sub class from Model, the following has to be
 * considered:
 *  <ul>
 *    <li>
 *      The model(...) member must be overloaded to compute the
 *      model's predictions. For differentiable models, the
 *      modelDerivative members should be overloaded, too.
 *    </li>
 *    <li>
 *      The members inputDimension and outputDimension must be set
 *      in the constructor. Further, the parameter vector has to
 *      be resized accordingly.
 *    </li>
 *    <li>
 *      If the model uses an internal parameter representation
 *      which differs from the parameter vector, setParameter(...)
 *      should be overloaded in order to keep the parameter values
 *      consistent with the parameter member.
 *    </li>
 *  </ul>
 *
 */
class Model
{
public:
	//! Constructor
	Model();

	//! Destructor
	virtual ~Model();


	//===========================================================================
	/*!
	*  \brief Returns the model's answer #output on the stimulus #input.
	*
	*  This method calculates the output of the model depending on the
	*  #input. The arrays #input and #output can either be one- or
	*  two-dimensional, depending on whether one or many patterns should
	*  be processed. The number of elements in the last dimension of
	*  these arrays must fit the #inputDimension and #outputDimension, in
	*  case of two-dimensional input, the number of elements in the first
	*  dimension equals the number of patterns. The method is pure
	*  virtual as it has to be implemented by the different models.
	* 
	*      \param  input Vector of input values.
	*      \param  output Vector of output values.
	*      \return None.
	*
	*/
	virtual void model(const Array<double>& input, Array<double>& output) = 0;

	//! Do a model evaluation on a const object
	inline void model(const Array<double>& input, Array<double> &output) const
	{
		Model* pT = const_cast<Model*>(this);
		pT->model(input, output);
	}

	//===========================================================================
	/*!
	*  \brief Calculates the derivative of the model
	*  output with respect to the parameters.
	*
	*  This method performs the calculation of the model output with
	*  respect to the model parameters. Sometimes this calculation can
	*  only be performed based on a single input pattern; therefore
	*  #input must be a onedimensional array. As a default
	*  implementation, the derivatives are estimated numerically by
	*  using small variations epsilon:
	* 
	*  \f[\frac{\partial f_j(x, p)}{\partial p_i} = 
	*      \frac{f_j(x, p + \epsilon e_i) - f_j(x, p)}{\epsilon} 
	*      + O(\epsilon) \f]
	*
	*  However, to improve the speed, it is adviserable to overload this
	*  method by a more sophisticated calculation, depending on the
	*  method #model. Nevertheless, this default implementation can be
	*  used to check new implementations of the derivative of the model
	*  with respect to the parameters.
	* 
	*      \param  input Vector of input values.
	*      \param  derivative Matrix of partial derivatives.
	*
	*/
	virtual void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! Compute the model derivative on a const object
	inline void modelDerivative(const Array<double>& input, Array<double>& derivative) const
	{
		Model* pT = const_cast<Model*>(this);
		pT->modelDerivative(input, derivative);
	}

	//===========================================================================
	/*!
	*  \brief Calculates the model output and the derivative of the model
	*  output with respect to the parameters.
	*
	*  This method performs the calculation of the model output with
	*  respect to the model parameters. Sometimes this calculation can
	*  only be performed based on a single input pattern; therefore
	*  #input must be a onedimensional array. As a default
	*  implementation, the derivatives are estimated numerically by
	*  using small variations epsilon:
	* 
	*  \f[\frac{\partial f_j(x, p)}{\partial p_i} = 
	*      \frac{f_j(x, p + \epsilon e_i) - f_j(x, p)}{\epsilon} 
	*      + O(\epsilon) \f]
	*
	*  However, to improve the speed, it is adviserable to overload this
	*  method by a more sophisticated calculation, depending on the
	*  method #model. Nevertheless, this default implementation can be
	*  used to check new implementations of the derivative of the model
	*  with respect to the parameters.
	* 
	*      \param  input Vector of input values.
	*      \param  output Vector of model output.
	*      \param  derivative Matrix of partial derivatives.
	*
	*/
	virtual void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative);

	//! Compute the model derivative on a const object
	inline void modelDerivative(const Array<double>& input, Array<double>& output, Array<double>& derivative) const
	{
		Model* pT = const_cast<Model*>(this);
		pT->modelDerivative(input, output, derivative);
	}

	//===========================================================================
	/*!
	*  \brief Calculates the model's general derivative.
	*
	*   After a call of #model, #generalDerivative calculates the
	*   gradient \f$\sum_i c_i\, \frac{\partial}{\partial p^j}
	*   f^i_p(x)\f$ and stores it in \em derivative. Here \f$x\f$ is
	*   some input pattern, and \f$p\f$ denotes the parameter vector of
	*   the model and \f$p^j\f$ its \f$p\f$th component. The index
	*   \f$i\f$ indicates the \f$i\f$th output. The coefficients
	*   \f$c_i\f$ can be chosen freely.
	*
	*   The point of this function is that for many models, the complexity
	*   of calculating this linear combination of gradients is the same as
	*   of calculating a single output gradient 
	* 
	*   \f[
	*   \frac{\partial}{\partial p} f^i_p(x)
	*   \f]
	*
	*   Via the chain rule, the gradient of _any_
	*   (e.g., error) functional can be calculated.
	*
	*
	*   \em Example \em 1: Let \f$c_i=f^i_p(x)-t^i\f$, then 
	*
	*   \f[
	*   \sum_i c_i\, \frac{\partial}{\partial p^j} f^i_p(x) = \frac{\partial}{\partial p^j}\, \Big[ \frac{1}{2}\, \sum_i(f^i_p(x)-t^i)^2 \Big]
	*   \f] 
	*
	*   is the parameter gradient of an MSE with respect to target
	*   values \f$t^i\f$ As above, the index \f$i\f$ runs over the
	*   outputs.
	*
	*
	*   \em Example \em 2: Let \f$c_i=\delta_{il}\f$, then the 
	*   gradients of only the \f$l\f$th output will be calculated: 
	*
	*   \f[
	*   \frac{\partial}{\partial p^j} f^l_p(x)
	*   \f]
	*
	*   and
	* 
	*   \f[
	*   \frac{\partial}{\partial x} f^l_p(x)
	*   \f]
	* 
	*   The default implementation is the computationally inefficient
	*   usage of modelDerivative.
	*
	*      \param  input the corefficients \f$c_i\f$ in the formula above.
	*      \param  coefficient the corefficients \f$c_i\f$ in the formula above.
	*      \param  derivative the calculated model's general derivative.
	*      \return None.
	*
	*  \author  M. Toussaint, C. Igel
	*  \date    2002
	*
	*/
	virtual void generalDerivative(const Array<double>& input, const Array<double>& coefficient, Array<double>& derivative);

	//! general derivative for a const object
	inline void generalDerivative(const Array<double>& input, const Array<double>& coefficient, Array<double>& derivative) const
	{
		Model* pT = const_cast<Model*>(this);
		pT->generalDerivative(input, coefficient, derivative);
	}

	/*!
	*  \brief check whether the parameters define a feasible model
	*
	*  The default implementation returns true, that is,
	*  every parameter configuration is considered feasible
	*  and unconstrained optimization is applicable.
	*  It is the Optimizer's responsibility to check the
	*  isFeasible() flag.
	* 
	*     \return true if the model is feasible, false otherwise
	*
	*  \author  T. Glasmachers
	*  \date    2006
	*
	*/
	virtual bool isFeasible();

	//! check whether the parameters define a feasible model on a const object
	inline bool isFeasible() const
	{
		Model* pT = const_cast<Model*>(this);
		return pT->isFeasible();
	}

	//! Returns the dimension of the model input.
	const inline unsigned int getInputDimension() const
	{
		return inputDimension;
	}

	//! Returns the dimension of the model output.
	const inline unsigned int getOutputDimension() const
	{
		return outputDimension;
	}

	//! Returns the number of optimizable model parameters,
	//! i.e. the dimension of the parameter array.
	const inline unsigned int getParameterDimension() const
	{
		SIZE_CHECK(parameter.ndim() == 1);
		return parameter.dim(0);
	}

	//! Returns a specific model parameter.
	virtual double getParameter(unsigned int index) const;

	//! Modifies a specific model parameter.
	virtual void setParameter(unsigned int index, double value);

	//===========================================================================
	/*!
	 *  \brief Sets the value of #epsilon.
	 *
	 *  The value of #epsilon is utilized for a numeric estimation of the
	 *  derivative of the error, with respect to the model parameters.
	 *  #epsilon should be choosen as a small value. The default value
	 *  is #epsilon = 1e-8.
	 *
	 *      \param  epsilon New value for epsilon.
	 *      \return None.
	 *
	 *  \author  C. Igel
	 *  \date    1999
	 *
	 *  \par Changes
	 *      none
	 *
	 *  \par Status
	 *      stable
	 *
	 */
	inline void setEpsilon(double eps)
	{
		epsilon = eps;
	};

	//!
	//! \brief Copy the model
	//!
	//! \par
	//! The Clone method first calls the virtual CloneI()
	//! method and then copies its parameters to the new model.
	//! Per definition, this is sufficient to obtain a copy of
	//! the model. The method may return NULL if the type of
	//! model does not support cloning.
	//!
	//! \author  T. Glasmachers
	//! \date    2008
	//!
	inline Model* Clone()
	{
		Model* ret = CloneI();
		if (ret == NULL) return NULL;
		int p, pc = getParameterDimension();
		for (p=0; p<pc; p++) ret->setParameter(p, getParameter(p));
		return ret;
	}

protected:
	//!
	//! \brief Clone the class of the model
	//!
	//! \par
	//! This method returns a new object of the same type
	//! like the current model. In most cases it is sufficient
	//! to return new MyModel(). However, complex models may
	//! manage additional information beyond their parameter
	//! vector. These information must be copied by CloneI in
	//! order to make Clone work correctly. If cloning fails,
	//! the method is required to return a NULL pointer.
	//!
	//! \author  T. Glasmachers
	//! \date    2008
	//!
	virtual Model* CloneI()
	{
		return NULL;
	}

	//! Model parameter vector.
	//! #parameter is a 1-dimensional array with the number
	//! of elements equal to the number of free parameters of
	//! the model (i.e., equal to the number of weights of a
	//! neural network).
	Array<double> parameter;

	//! Dimension of the data accepted by the model as input.
	unsigned int inputDimension;

	//! Dimension of the model output per single input pattern.
	unsigned int outputDimension;

	//! Precision parameter for the numerical error gradient approximation.
	double epsilon;

public:
	//! \brief Read the model parameters from a stream.
	//!
	//! The model parameters are read from a text stream.
	//! The single numbers are separated with a single
	//! space character. The line break CR/LF is used as
	//! an end marker. The setParameter member is used
	//! to set the parameter values in order to support
	//! models that override the parameter manegement.
  //!
  //! This virtual method is called by the stream operator.
	//!
	//! \author  C. Igel
	//! \date    2009
	//!
	virtual void read(std::istream& is);

	//! \brief Write the model parameters to a stream.
	//!
	//! The model parameters are written to a text stream.
	//! The single numbers are separated with a single
	//! space character. The line break CR/LF is used as
	//! an end marker.
	//!
  //! This virtual method is called by the stream operator.
	//!
	//! \author  C. Igel
	//! \date    2009
	//!
	virtual void write(std::ostream& os) const;

	//! \brief Read the model parameters from a stream.
	//!
	//! The model parameters are read from a text stream.
	//! The single numbers are separated with a single
	//! space character. The line break CR/LF is used as
	//! an end marker. The setParameter member is used
	//! to set the parameter values in order to support
	//! models that override the parameter manegement.
	//!
	//! \author  T. Glasmachers
	//! \date    2006
	//!
	friend std::istream& operator >> (std::istream& is, Model& model);

	//! \brief Write the model parameters to a stream.
	//!
	//! The model parameters are written to a text stream.
	//! The single numbers are separated with a single
	//! space character. The line break CR/LF is used as
	//! an end marker.
	//!
	//! \author  T. Glasmachers
	//! \date    2006
	//!
	friend std::ostream& operator << (std::ostream& os, const Model& model);
	//! Read the model parameters from a file.
	//! \author  T. Glasmachers
	//! \date    2006
	bool load(const char* filename);

	//! Write the model parameters to a file.
	//! \author  T. Glasmachers
	//! \date    2006
	bool save(const char* filename);
};


#endif

