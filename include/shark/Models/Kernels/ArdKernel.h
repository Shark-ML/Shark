//===========================================================================
/*!
 * 
 *
 * \brief       Gaussian automatic relevance detection (ARD) kernel
 * 
 * 
 *
 * \author      T.Glasmachers, O. Krause, M. Tuma
 * \date        2010-2012
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
 * 
 * <BR><HR>
 * This file is part of Shark.
 * <http://image.diku.dk/shark/>
 * 
 * Shark is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Shark is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with Shark.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//===========================================================================

#ifndef SHARK_MODELS_KERNELS_GAUSSIAN_ARD_KERNEL_H
#define SHARK_MODELS_KERNELS_GAUSSIAN_ARD_KERNEL_H


#include <shark/Models/Kernels/AbstractKernelFunction.h>
namespace shark {


/// \brief Automatic relevance detection kernel for unconstrained parameter optimization
///
/// The automatic relevance detection (ARD) kernel is a general Gaussian kernel with
/// diagonal covariance matrix:
/// \f$ k(x, z) = \exp(-\sum_i \gamma_i  (x_i - z_i)^2) \f$.
/// The ARD kernel holds one real-valued parameter \f$ \gamma_i \f$ per input dimension.
/// The parameters \f$ p_i \f$ are encoded as \f$ p_i^2 = \gamma_i \f$, allowing for unconstrained
/// optimization. Here, the exposed/visible parameters are squared before being used in the
/// actual computations, because the otherwise Shark-canonical exponential encoding proved
/// a bit unstable, that is, leading to too big step sizes, in preliminary experiments.
///
/// Note that, like all or most models/kernels designed for unconstrained optimization, the
/// argument to the constructor corresponds to the value of the true weights, while the set
/// and get methods for the parameter vector set the parameterized values and not the true weights.
///
/// \todo slow default implementation. Use BLAS3 calls to make things faster
template<class InputType=RealVector>
class ARDKernelUnconstrained : public AbstractKernelFunction<InputType>
{
private:
	typedef AbstractKernelFunction<InputType> base_type;

	struct InternalState: public State{
		RealMatrix kxy;

		void resize(std::size_t sizeX1,std::size_t sizeX2){
			kxy.resize(sizeX1,sizeX2);
		}
	};
public:
	typedef typename base_type::BatchInputType BatchInputType;
	typedef typename base_type::ConstInputReference ConstInputReference;
	typedef typename base_type::ConstBatchInputReference ConstBatchInputReference;

	/// Constructor
	/// \param dim input dimension
	/// \param gamma_init initial gamma value for all dimensions (true value, used as passed into ctor)
	ARDKernelUnconstrained(unsigned int dim, double gamma_init = 1.0){
		SHARK_CHECK( gamma_init > 0, "[ARDKernelUnconstrained::ARDKernelUnconstrained] Expected positive weight.");

		//init abstract model's informational flags
		this->m_features|=base_type::HAS_FIRST_PARAMETER_DERIVATIVE;
		this->m_features|=base_type::HAS_FIRST_INPUT_DERIVATIVE;
		this->m_features|=base_type::IS_NORMALIZED;

		//initialize self
		m_inputDimensions = dim;
		m_gammas.resize(m_inputDimensions);
		m_params.resize(m_inputDimensions);
		double sqrt_gamma = std::sqrt( gamma_init );
		for ( unsigned int i=0; i<m_inputDimensions; i++ ){
			m_gammas(i) = gamma_init;
			m_params(i) = sqrt_gamma;
		}
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "ARDKernelUnconstrained"; }

	RealVector parameterVector() const{
		return m_params;
	}
	void setParameterVector(RealVector const& newParameters){
		SIZE_CHECK(newParameters.size() == m_inputDimensions);
		m_params = newParameters;
		noalias(m_gammas) = sqr(m_params);
	}
	std::size_t numberOfParameters() const{
		return m_inputDimensions;
	}

	///\brief creates the internal state of the kernel
	boost::shared_ptr<State> createState()const{
		return boost::shared_ptr<State>(new InternalState());
	}

	/// convenience methods for setting/getting the actual gamma values
	RealVector gammaVector() const{
		return m_gammas;
	}
	void setGammaVector( RealVector const& newGammas ) {
#ifndef DNDEBUG
		SIZE_CHECK( newGammas.size() == m_inputDimensions );
		for ( unsigned int i=0; i<m_inputDimensions; i++ ) {
			RANGE_CHECK( newGammas(i) > 0 );
		}
#endif
		m_gammas = newGammas;
		noalias(m_params) = sqrt(m_gammas);
	}

	/// \brief evaluates \f$ k(x,z)\f$
	///
	/// ARD kernel evaluation
	/// \f[ k(x, z) = \exp(-\sum_i \gamma_i  (x_i - z_i)^2) \f]
	double eval(ConstInputReference x1, ConstInputReference x2) const{
		SIZE_CHECK(x1.size() == x2.size());
		SIZE_CHECK(x1.size() == m_inputDimensions);
		double dmnorm2 = diagonalMahalanobisDistanceSqr(x1, x2, m_gammas);
		return std::exp(-dmnorm2);
	}

	/// \brief evaluates \f$ k(x,z)\f$ for a whole batch
	///
	/// ARD kernel evaluation
	/// \f[ k(x, z) = \exp(-\sum_i \gamma_i  (x_i - z_i)^2) \f]
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(batchX1.size2() == m_inputDimensions);

		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();

		ensure_size(result,sizeX1,sizeX2);
		//todo: implement fast version of diagonalMahalanobisDistanceSqr for matrices
		for(std::size_t i = 0; i != sizeX1; ++i){
			for(std::size_t j = 0; j != sizeX2; ++j){
				double dmnorm2 = diagonalMahalanobisDistanceSqr(row(batchX1,i), row(batchX2,j), m_gammas);
				result(i,j)=std::exp(-dmnorm2);
			}
		}
	}

	/// \brief evaluates \f$ k(x,z)\f$ for a whole batch
	///
	/// ARD kernel evaluation
	/// \f[ k(x, z) = \exp(-\sum_i \gamma_i  (x_i - z_i)^2) \f]
	void eval(ConstBatchInputReference batchX1, ConstBatchInputReference batchX2, RealMatrix& result, State& state) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(batchX1.size2() == m_inputDimensions);

		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();

		InternalState& s = state.toState<InternalState>();
		s.resize(sizeX1,sizeX2);

		ensure_size(result,sizeX1,sizeX2);
		//todo: implement fast version of diagonalMahalanobisDistanceSqr for matrices
		for(std::size_t i = 0; i != sizeX1; ++i){
			for(std::size_t j = 0; j != sizeX2; ++j){
				double dmnorm2 = diagonalMahalanobisDistanceSqr(row(batchX1,i), row(batchX2,j), m_gammas);
				result(i,j) = std::exp(-dmnorm2);
				s.kxy(i,j) = result(i,j);
			}
		}
	}

	/// \brief evaluates \f$ \frac {\partial k(x,z)}{\partial \sqrt{\gamma_i}}\f$ weighted over a whole batch
	///
	/// Since the ARD kernel is parametrized for unconstrained optimization, we return
	/// the derivative w.r.t. the parameters \f$ p_i \f$, where \f$ p_i^2 = \gamma_i \f$.
	///
	/// \f[ \frac {\partial k(x,z)}{\partial p_i} = -2 p_i (x_i - z_i)^2 \cdot k(x,z) \f]
	void weightedParameterDerivative(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficients,
		State const& state,
		RealVector& gradient
	) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(batchX1.size2() == m_inputDimensions);

		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();

		ensure_size(gradient, m_inputDimensions );
		gradient.clear();
		InternalState const& s = state.toState<InternalState>();

		for(std::size_t i = 0; i != sizeX1; ++i){
			for(std::size_t j = 0; j != sizeX2; ++j){
				double coeff = coefficients(i,j) * s.kxy(i,j);
				gradient += coeff * m_params * sqr(row(batchX1,i)-row(batchX2,j));
			}
		}
		gradient *= -2;
 	}

	/// \brief evaluates \f$ \frac {\partial k(x,z)}{\partial x}\f$
	///
	/// first derivative of ARD kernel wrt the first input pattern
	/// \f[ \frac {\partial k(x,z)}{\partial x} = -2 \gamma_i \left( x_i - z_i \right)\cdot k(x,z) \f]
	void weightedInputDerivative(
		ConstBatchInputReference batchX1,
		ConstBatchInputReference batchX2,
		RealMatrix const& coefficientsX2,
		State const& state,
		BatchInputType& gradient
	) const{
		SIZE_CHECK(batchX1.size2() == batchX2.size2());
		SIZE_CHECK(batchX1.size2() == m_inputDimensions);

		std::size_t sizeX1 = batchX1.size1();
		std::size_t sizeX2 = batchX2.size1();

		InternalState const& s = state.toState<InternalState>();
		ensure_size(gradient, sizeX1, m_inputDimensions );
		gradient.clear();

		for(std::size_t i = 0; i != sizeX1; ++i){
			for(std::size_t j = 0; j != sizeX2; ++j){
				double coeff = coefficientsX2(i,j) * s.kxy(i,j);
				row(gradient,i) += coeff * m_gammas * (row(batchX1,i)-row(batchX2,j));
			}
		}
		gradient *= -2.0;
	}

	void read(InArchive& ar){
		ar >> m_gammas;
		ar >> m_inputDimensions;
		m_params.resize(m_inputDimensions);
		noalias(m_params) = sqrt(m_gammas);
	}

	void write(OutArchive& ar) const{
		ar << m_gammas;
		ar << m_inputDimensions;
	}

protected:
	RealVector m_gammas;				///< kernel bandwidth parameters, one for each input dimension. squares of m_params.
	RealVector m_params;				///< parameters as seen by the external optimizer (for unconstrained optimization). can be negative.
	std::size_t m_inputDimensions;		///< how many input dimensions = how many bandwidth parameters
};

typedef ARDKernelUnconstrained<> DenseARDKernel;
typedef ARDKernelUnconstrained<CompressedRealVector> CompressedARDKernel;
typedef ARDKernelUnconstrained<ConstRealVectorRange> DenseARDMklKernel;

}
#endif
