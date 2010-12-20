
//===========================================================================
/*!
*  \file KernelFunction.h
*
*  \brief This file contains the definition of a kernel
*         function as well as basic examples of kernel functions.
*
*  \author  T. Glasmachers
*  \date    2005
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

#ifndef _KernelFunction_H_
#define _KernelFunction_H_


#include <ReClaM/Model.h>
#include <vector>


/*!
 *
 *  \brief Definition of a kernel function as a ReClaM model
 *
 *  The interpretation of a kernel function
 *  as a ReClaM Model involves some complications. We have
 *  to cope with the problem that the Model usually produces
 *  one output per input, while the KernelFunction produces
 *  one output for a pair of inputs. Nevertheless, we want the
 *  KernelFunction to be a parameterized ReClaM Model for the
 *  sake of optimization.
 *  For simplicity, the input data are required to be matrices
 *  composed of exactly two columns for KernelFunction Models.
 *  For this reason, the KernelFunction provides an additional
 *  interface through its virtual members #eval and
 *  #evalDerivative that should be overriden. The inherited
 *  members #model and #modelDerivative redirect to this
 *  interface.
 *
 *  The #modelDerivative member considerably differs from the
 *  standard behavior as it requires a two dimensional input
 *  array (holding two input patterns). Usually, only one
 *  input pattern is allowed for derivative computations.
 *  On the other hand, the #derivative array is one dimensional,
 *  indexed only by the kernel parameters.
 *
 *  According to the differences between #KernelFunction and
 *  #Model it is advisable to use the #eval and #evalDerivative
 *  interface whenever it is clear that the object under
 *  consideration is a #KernelFunction.
 *
 */
class KernelFunction : public Model
{
public:
	//! Constructor
	KernelFunction();

	//! Destructor
	virtual ~KernelFunction();


	//! Evaluates the kernel function on a const object.
	virtual double eval(const Array<double>& x1, const Array<double>& x2) const = 0;

	//! Evaluates the kernel function and computes
	//! its derivatives w.r.t. the kernel parameters.
	virtual double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	//! Kernel evaluation in the form of an operator
	inline double operator()(const Array<double>& x1, const Array<double>& x2)
	{
		return eval(x1, x2);
	}

	//! The Model behaviour of the KernelFunction is to interpret
	//! the input as a matrix of two vectors and to compute the
	//! kernel value on them.
	void model(const Array<double>& input, Array<double> &output);

	//! Same as #model, additionally the derivatives w.r.t. all kernel
	//! parameters are computed.
	void modelDerivative(const Array<double>& input, Array<double>& derivative);

	//! Same as #model, additionally the derivatives w.r.t. all kernel
	//! parameters are computed.
	void modelDerivative(const Array<double>& input, Array<double> &output, Array<double>& derivative);

	friend class C_SVM;
};

class SparseKernelFunction : public KernelFunction
{
public: 
	SparseKernelFunction();
	virtual ~SparseKernelFunction();
	
	virtual void constructSparseFromLibSVM( const char* filename, long train, long test = -1 );
	
	double eval(const Array<double>& x1, const Array<double>& x2) const; //will throw an exeption for this base class
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const; //will throw an exeption for this base class
	
	//! access to the training targets as a constant array
	inline const Array<double> & getTrainingTarget() const
	{
		return trainingTarget;
	}

	//! access to the test targets as a constant array
	inline const Array<double> & getTestTarget() const
	{
		return testTarget;
	}

	inline const long getNoofRows() const
	{
		return noof_rows;
	}

	inline void printSparsenessRatios() const
	{
		std::cout << "sparsenessRatioTrain = " << sparsenessRatioTrain 
				  << ", sparsenessRatioTest = " << sparsenessRatioTest << std::endl;
	}
	
protected:
	
	struct tIndexValuePair
	{
		long sparseIndex;
		double sparseValue;
	};
	
	long noof_rows;
	bool wasInitialized;
	tIndexValuePair ** sparseData;
	Array<double> trainingTarget;
	Array<double> testTarget;
	
	double sparsenessRatioTrain, sparsenessRatioTest;
};


//! \brief Linear Kernel, parameter free
class LinearKernel : public KernelFunction
{
public:
	LinearKernel();
	~LinearKernel();


	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;
};

//! \brief linear kernel optimized for sparse data. parameter free
class SparseLinearKernel : public SparseKernelFunction
{
public:
	SparseLinearKernel();
	~SparseLinearKernel();

	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;
};


//! \brief Polynomial Kernel
class PolynomialKernel : public KernelFunction
{
public:
	PolynomialKernel(int degree, double offset);
	~PolynomialKernel();


	double eval(const Array<double>& x1, const Array<double>& x2) const;

	// TODO (later...)
	// Only fill in the derivative w.r.t. the offset?
	// double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	void setParameter(unsigned int index, double value);

	bool isFeasible();
};


/*!
 *
 *  \brief Definition of the RBF Gaussian kernel
 *
 *  A special but very important type of kernel is the Gaussian
 *  normal distribution density kernel
 *  \f[
 *  exp(-\gamma \|x_1 - x_2\|^2)
 *  \f]
 *  It has a single parameter \f$\gamma > 0\f$ controlling the kernel
 *  width \f$\sigma = \sqrt{2 / \gamma}\f$.
 */
class RBFKernel : public KernelFunction
{
public:
	RBFKernel(double gamma);
	~RBFKernel();


	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	bool isFeasible();

	double getSigma();
	void setSigma(double sigma);
};


class SparseRBFKernel : public SparseKernelFunction
{
public:
	SparseRBFKernel(double gamma);
	~SparseRBFKernel();

	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;
	
	bool isFeasible();
	double getSigma();
	void setSigma(double sigma);
};


//! \brief Gaussian rbf kernel, with density normalization
class NormalizedRBFKernel : public KernelFunction
{
public:
	NormalizedRBFKernel();
	NormalizedRBFKernel(double s);


	void setSigma(double s);
	double eval(const Array<double> &x,  const Array<double> &z) const;
	double evalDerivative(const Array<double> &x,  const Array<double> &z, Array<double>& derivative) const;
};


/*!
 *
 *  \brief Normalized version of a kernel function
 *
 *  For a positive definite kernel k, the normalized kernel
 *  \f[
 *      \tilde k(x_1, x_2) := \frac{k(x_1, x_2)}{\sqrt{k(x_1, x_1) \cdot k(x_2, x_2)}}
 *  \f]
 *  is again a positive definite kernel function.
 *
 */
class NormalizedKernel : public KernelFunction
{
public:
	NormalizedKernel(KernelFunction* base);
	~NormalizedKernel();


	void setParameter(unsigned int index, double value);
	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	bool isFeasible();

protected:
	//! kernel to normalize
	KernelFunction* baseKernel;
};


/*!
 *
 *  \brief Weighted sum of kernel functions
 *
 *  For a set of positive definite kernels \f$ k_1, \dots, k_n \f$
 *  with positive coeffitients \f$ w_1, \dots, w_n \f$ the sum
 *  \f[
 *      \tilde k(x_1, x_2) := \sum_{i=1}^{n} w_i \cdot k_i(x_1, x_2)
 *  \f]
 *  is again a positive definite kernel function.
 *  Internally, the weights are represented as
 *  \f$ w_i = \exp(\xi_i) \f$
 *  to allow for unconstrained optimization. Further, one
 *  parameter is removed by fixing the sum of the weights to
 *  \f$ \sum_{i=1}^{n} w_i = 1 \f$.
 *
 */
class WeightedSumKernel : public KernelFunction
{
public:
	WeightedSumKernel(const std::vector<KernelFunction*>& base);
	~WeightedSumKernel();


	void setParameter(unsigned int index, double value);
	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	bool isFeasible();

protected:
	//! kernels to combine
	std::vector<KernelFunction*> baseKernel;

	//! current weighting factors
	std::vector<double> weight;

	//! sum over all weighting factors
	double weightsum;
};


/*!
 *
 *  \brief Weighted sum of kernel functions
 *
 *  For a set of positive definite kernels \f$ k_1, \dots, k_n \f$
 *  with positive coeffitients \f$ w_1, \dots, w_n \f$ the sum
 *  \f[
 *      \tilde k(x_1, x_2) := \sum_{i=1}^{n} w_i \cdot k_i(x_1, x_2)
 *  \f]
 *  is again a positive definite kernel function.
 *  Internally, the weights are represented as
 *  \f$ w_i = \exp(\xi_i) \f$
 *  to allow for unconstrained optimization. If it seems reasonable
 *  to get rid of kernel scaling please consider using the
 *  WeightedSumKernel class.
 *
 */
class WeightedSumKernel2 : public KernelFunction
{
public:
	WeightedSumKernel2(const std::vector<KernelFunction*>& base);
	~WeightedSumKernel2();


	void setParameter(unsigned int index, double value);
	double eval(const Array<double>& x1, const Array<double>& x2) const;
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	bool isFeasible();

protected:
	std::vector<KernelFunction*> baseKernel;

	std::vector<double> weight;

	double weightsum;
};


/*!
 *  \brief kernel based on prototype vectors
 *
 *  \par
 *  The PrototypeKernel class provides a kernel function on an
 *  arbitrary finite set. The set is (w.r.o.g.) assumed to be
 *  given as the first N natural numbers {0, ..., N-1}.
 *  There is one prototype vector \f$ v_i \f$ for each element.
 *  Then the kernel is computed as \f$ k(i, j) = v_i^T v_j \f$.
 */
class PrototypeKernel : public KernelFunction
{
public:
	//! Constructor
	//! \param  setsize       size of the discrete set
	//! \param  prototypeDim  dimension of the prototype vectors; if zero this coincides with setsize
	PrototypeKernel(unsigned int setsize, unsigned int prototypeDim = 0);

	//! Destructor
	~PrototypeKernel();


	//! evaluate the kernel
	double eval(const Array<double>& x1, const Array<double>& x2) const;

	//! evaluate the kernel and compute the derivative w.r.t. its parameters
	double evalDerivative(const Array<double>& x1, const Array<double>& x2, Array<double>& derivative) const;

	//! read access to the n-th prototype vector
	void getPrototype(unsigned int n, Array<double>& vec);

	//! write access to the n-th prototype vector
	void setPrototype(unsigned int n, const Array<double>& vec);

protected:
	unsigned int m_setsize;
	unsigned int m_prototypeDim;
};


/*!
 *  \brief Joint kernel for input and label space
 *
 *  \par
 *  The JointKernelFunction class basically computes a kernel
 *  on \f$(X \times Y)\f$ where X is the input space and Y is
 *  the output space of a model. The computation is done as
 *  \f[
 *      k((x_1, y_1), (x_2, y_2)) = k_X(x_1, x_2) \cdot k_Y(y_1, y_2) \enspace.
 *  \f]
 *  The JointKernelFunction class holds the parameters of both
 *  (input and label) kernel objects.
 */
class JointKernelFunction : public Model
{
public:
	//! Constructor
	JointKernelFunction(KernelFunction& inputkernel, KernelFunction& labelkernel);

	//! Destructor
	~JointKernelFunction();


	//! Evaluates the kernel function on a const object.
	double eval(const Array<double>& x1, const Array<double>& y1, const Array<double>& x2, const Array<double>& y2) const;

	//! Evaluates the kernel function and computes
	//! its derivatives w.r.t. the kernel parameters.
	double evalDerivative(const Array<double>& x1, const Array<double>& y1, const Array<double>& x2, const Array<double>& y2, Array<double>& derivative) const;

	//! undefined; just throws an exception
	void model(const Array<double>& input, Array<double> &output);

	//! set a parameter of either input or label kernel
	void setParameter(unsigned int index, double value);

protected:
	//! input space kernel
	KernelFunction& m_inputkernel;

	//! label/output space kernel
	KernelFunction& m_labelkernel;
};


#endif
