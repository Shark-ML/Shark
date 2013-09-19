//===========================================================================
/*!
 *  \brief A kernel expansion with support of missing features
 *
 *  \author  B. Li
 *  \date    2012
 *
 *  \par Copyright (c) 2012:
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
 *  Foundation; either version 3, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this library; if not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================
#include <shark/Data/Dataset.h>
#include <shark/Data/DataView.h>
#include <shark/Models/Kernels/EvalSkipMissingFeatures.h>
#include <shark/Models/Kernels/KernelExpansion.h>

#include <boost/foreach.hpp>
#include <boost/optional/optional.hpp>

namespace shark {

/// Kernel expansion with missing features support
template<class InputType>
class MissingFeaturesKernelExpansion : public KernelExpansion<InputType>
{
private:
	typedef KernelExpansion<InputType> Base;
public:
	typedef typename Base::KernelType KernelType;
	typedef typename Base::BatchInputType BatchInputType;
	typedef typename Base::BatchOutputType BatchOutputType;
	/// Constructors from the base class
	///@{
	MissingFeaturesKernelExpansion(){}


	MissingFeaturesKernelExpansion(KernelType* kernel)
	: Base(kernel)
	{}

	MissingFeaturesKernelExpansion(KernelType* kernel, Data<InputType> const& basis, bool offset)
	: Base(kernel, basis, offset, 1u)
	{}
	///@}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "MissingFeaturesKernelExpansion"; }

	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new EmptyState());
	}

	/// Override eval(...) in the base class
	virtual void eval(BatchInputType const& patterns, BatchOutputType& outputs)const{
		SHARK_ASSERT(Base::mep_kernel);
		SIZE_CHECK(Base::m_alpha.size1() > 0u);
		
		//Todo: i am too lazy to us iterated loops in this function.
		//so i am using a DataView to have O(1) random access lookup. but this is not needed!
		DataView<Data<InputType> const > indexedBasis(Base::m_basis);
		
		ensureSize(outputs,size(patterns),Base::m_outputs);
		if (Base::hasOffset())
				noalias(outputs) = repeat(Base::m_b,size(patterns));
			else
				zero(outputs);
		
		for(std::size_t p = 0; p != size(patterns); ++p){


			// Calculate scaling coefficient for the 'pattern'
			const double patternNorm = computeNorm(column(Base::m_alpha, 0), m_scalingCoefficients, get(patterns,p));
			const double patternSc = patternNorm / m_classifierNorm;

			// Do normal classification except that we use kernel which supports inputs with Missing features
			//TODO: evaluate k for all i and replace the += with a matrix-vector operation. 
			//better: do this for all p and i and go matrix-matrix-multiplication
			for (std::size_t i = 0; i != indexedBasis.size(); ++i){
				const double k = evalSkipMissingFeatures(
					*Base::mep_kernel,
					indexedBasis[i],
					get(patterns,p)) / m_scalingCoefficients[i] / patternSc;
				noalias(row(outputs,p)) += k * row(Base::m_alpha, i);
				
			}
		}
	}
	void eval(BatchInputType const& patterns, BatchOutputType& outputs, State & state)const{
		eval(patterns, outputs);
	}

	/// Calculate norm of classifier, i.e., ||w||
	///
	/// formula:
	/// \f$ \sum_{i,j=1}^{n}\alpha_i\frac{y_i}{s_i}K\left(x_i,x_j)\right)\frac{y_j}{s_j}\alpha_j \f$
	/// where \f$ s_i \f$ is scaling coefficient, and \f$ K \f$ is kernel function,
	/// \f$ K\left(x_i,x_j)\right) \f$ is taken only over features that are valid for both \f$ x_i \f$ and \f$ x_j \f$
	template<class InputTypeT>
	double computeNorm(
		const RealVector& alpha,
		const RealVector& scalingCoefficient,
		InputTypeT const& missingness
	) const{
		SHARK_ASSERT(Base::mep_kernel);
		SIZE_CHECK(alpha.size() == scalingCoefficient.size());
		SIZE_CHECK(Base::m_basis.numberOfElements() == alpha.size());

		// Calculate ||w||^2
		double norm_sqr = 0.0;
		
		//Todo: i am too lazy to use iterated loops in this function.
		//so i am using a DataView to have O(1) random access lookup. but this is not needed!
		DataView<Data<InputType> const > indexedBasis(Base::m_basis);

		for (std::size_t i = 0; i < alpha.size(); ++i){
			for (std::size_t j = 0; j < alpha.size(); ++j){
				const double evalResult = evalSkipMissingFeatures(
					*Base::mep_kernel,
					indexedBasis[i],
					indexedBasis[j],
					missingness);
				// Note that in Shark solver, we do axis flip by substituting \alpha with y \times \alpha
				norm_sqr += evalResult * alpha(i) * alpha(j) / scalingCoefficient(i) / scalingCoefficient(j);
			}
		}

		// Return ||w||
		return std::sqrt(norm_sqr);
	}
	
	double computeNorm(
		const RealVector& alpha,
		const RealVector& scalingCoefficient
	) const{
		SHARK_ASSERT(Base::mep_kernel);
		SIZE_CHECK(alpha.size() == scalingCoefficient.size());
		SIZE_CHECK(Base::m_basis.numberOfElements() == alpha.size());
		
		//Todo: i am too lazy to us iterated loops in this function.
		//so i am using a DataView to have O(1) random access lookup. but this is not needed!
		DataView<Data<InputType> const > indexedBasis(Base::m_basis);

		// Calculate ||w||^2
		double norm_sqr = 0.0;
		
		for (std::size_t i = 0; i < alpha.size(); ++i){
			for (std::size_t j = 0; j < alpha.size(); ++j){
				const double evalResult = evalSkipMissingFeatures(
					*Base::mep_kernel,
					indexedBasis[i],
					indexedBasis[j]);
				// Note that in Shark solver, we do axis flip by substituting \alpha with y \times \alpha
				norm_sqr += evalResult * alpha(i) * alpha(j) / scalingCoefficient(i) / scalingCoefficient(j);
			}
		}

		// Return ||w||
		return std::sqrt(norm_sqr);
	}

	void setScalingCoefficients(const RealVector& scalingCoefficients)
	{
#if DEBUG
		BOOST_FOREACH(double v, scalingCoefficients)
		{
			SHARK_ASSERT(v > 0.0);
		}
#endif
		m_scalingCoefficients = scalingCoefficients;
	}

	void setClassifierNorm(double classifierNorm)
	{
		SHARK_ASSERT(classifierNorm > 0.0);
		m_classifierNorm = classifierNorm;
	}

protected:
	/// The scaling coefficients
	RealVector m_scalingCoefficients;

	/// The norm of classifier(w)
	double m_classifierNorm;
};

} // namespace shark {
