/*!
 *
 *
 * \brief       Kernel Target Alignment - a measure of alignment of a kernel Gram matrix with labels.
 * \file
 *
 *
 * \author      T. Glasmachers, O.Krause
 * \date        2010-2012
 *
 *
 * \par Copyright 1995-2017 Shark Development Team
 *
 * <BR><HR>
 * This file is part of Shark.
 * <http://shark-ml.org/>
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
#ifndef SHARK_OBJECTIVEFUNCTIONS_KERNELTARGETALIGNMENT_H
#define SHARK_OBJECTIVEFUNCTIONS_KERNELTARGETALIGNMENT_H

#include <shark/ObjectiveFunctions/AbstractObjectiveFunction.h>
#include <shark/Data/Dataset.h>
#include <shark/Data/Statistics.h>
#include <shark/Models/Kernels/AbstractKernelFunction.h>


namespace shark{
	
	
/// \defgroup kerneloptimization Kernel Optimization
/// \ingroup objfunctions
/// \ingroup kernels
/// \brief All kinds of objective functions to optimize kernel functions.
///

///  \brief Kernel Target Alignment - a measure of alignment of a kernel Gram matrix with labels.
///
///The Kernel Target Alignment (KTA) was originally proposed in the paper:<br/>
///<i>On Kernel-Target Alignment</i>. N. Cristianini, J. Shawe-Taylor,
///A. Elisseeff, J. Kandola. Innovations in Machine Learning, 2006.<br/>
///Here we provide a version with centering of the features as proposed
///in the paper:<br/>
///<i>Two-Stage Learning Kernel Algorithms</i>. C. Cortes, M. Mohri,
///A. Rostamizadeh. ICML 2010.<br/>
///
///The kernel target alignment is defined as
///where K is the kernel Gram matrix of the data and y is the vector of
///\f[ \hat A = \frac{\langle K, y y^T \rangle}{\sqrt{\langle K, K \rangle \cdot \langle y y^T, y y^T \rangle}} \f]
///+1/-1 valued labels. The outer product \f$y y^T\f$ corresponds to
///an ideal Gram matrix corresponding to a kernel that maps
///the two classes each to a single point, thus minimizing within-class
///distance for fixed inter-class distance. The inner products denote the
///Frobenius product of matrices:
///http://en.wikipedia.org/wiki/Matrix_multiplication#Frobenius_product
///
///In kernel-based learning, the kernel Gram matrix \f$K\f$ is of the form
///\f[ K_{i,j} = k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle \f]
///for a Mercer kernel function k and inputs \f$x_i, x_j\f$. In this
///version of the KTA we use centered feature vectors. Let
///\f[ \psi(x_i) = \phi(x_i) - \frac{1}{\ell} \sum_{j=1}^{\ell} \phi(x_j) \f]
///denote the centered feature vectors, then the centered Gram matrix \f$K^c\f$ is given by
///\f[ K^c_{i,j} = \langle \psi(x_i), \psi(x_j) \rangle = K_{i,j} - \frac{1}{\ell} \sum_{n=1}^\ell K_{i,n} + K_{j,n} + \frac{1}{\ell^2} \sum_{m,n=1}^\ell K_{n,m} \f]
///The alignment measure computed by this class is the exact same formula
///for \f$ \hat A \f$, but with \f$K^c\f$ plugged in in place of \f$K\f$.
///
///KTA measures the Frobenius inner product between a kernel Gram matrix
///and this ideal matrix. The interpretation is that KTA measures how
///well a given kernel fits a classification problem. The actual measure
///is invariant under kernel rescaling.
///In Shark, objective functions are minimized by convention. Therefore
///the negative alignment \f$- \hat A\f$ is implemented. The measure is
///extended for multi-class problems by using prototype vectors instead
///of scalar labels.
///
///The following properties of KTA are important from a model selection
///point of view: it is relatively fast and easy to compute, it is
///differentiable w.r.t. the kernel function, and it is independent of
///the actual classifier.
/// \ingroup kerneloptimization
template<class InputType = RealVector,class LabelType = unsigned int>
class KernelTargetAlignment : public AbstractObjectiveFunction< RealVector, double >
{
private:
	typedef typename Batch<LabelType>::type BatchLabelType;
public:
	/// \brief Construction of the Kernel Target Alignment (KTA) from a kernel object.
	KernelTargetAlignment(
		LabeledData<InputType, LabelType> const& dataset,
		AbstractKernelFunction<InputType>* kernel,
		bool centering = true
	){
		SHARK_RUNTIME_CHECK(kernel != NULL, "[KernelTargetAlignment] kernel must not be NULL");

		mep_kernel = kernel;

		m_features|=HAS_VALUE;
		m_features|=CAN_PROPOSE_STARTING_POINT;

		if(mep_kernel -> hasFirstParameterDerivative())
			m_features|=HAS_FIRST_DERIVATIVE;

		m_data = dataset;
		m_elements = dataset.numberOfElements();
		m_centering = centering;

		setupY(dataset.labels(), centering);
	}

	/// \brief From INameable: return the class name.
	std::string name() const
	{ return "KernelTargetAlignment"; }

	/// Return the current kernel parameters as a starting point for an optimization run.
	SearchPointType proposeStartingPoint() const {
		return  mep_kernel -> parameterVector();
	}


	std::size_t numberOfVariables()const{
		return mep_kernel->numberOfParameters();
	}

	/// \brief Evaluate the (centered, negative) Kernel Target Alignment (KTA).
	///
	/// See the class description for more details on this computation.
	double eval(RealVector const& input) const{
		mep_kernel->setParameterVector(input);

		return -evaluateKernelMatrix().error;
	}

	/// \brief Compute the derivative of the KTA as a function of the kernel parameters.
	///
	/// It holds:
	/// \f[ \langle K^c, K^c \rangle = \langle K,K \rangle  -2 \ell \langle k,k \rangle  + mk^2 \ell^2 \\
	///     (\langle  K^c, K^c  \rangle )'  = 2 \langle K,K' \rangle  -4\ell \langle k, \frac{1}{\ell} \sum_j K'_ij \rangle  +2 \ell^2 mk \sum_ij 1/(\ell^2) K'_ij \\
	///   = 2 \langle K,K' \rangle  -4 \langle k, \sum_j K'_ij \rangle + 2 mk \sum_ij K_ij \\
	///   = 2 \langle K,K' \rangle  -4 \langle k u^T, K' \rangle + 2 mk \langle  u u^T, K' \rangle \\
	///   = 2\langle K -2 k u^T + mk u u^T, K' \rangle ) \\
	///     \langle Y, K^c \rangle  = \langle Y, K \rangle  - 2 n \langle y, k \rangle  + n^2 my mk \\
	///     (\langle  Y  , K^c  \rangle )' =   \langle Y -2 y u^T + my u u^T, K'  \rangle \f]
	/// now the derivative is computed from this values in a second sweep over the data:
	/// we get:
	/// \f[ \hat A' = 1/\langle K^c,K^c \rangle ^{3/2} (\langle K^c,K^c \rangle  (\langle Y,K^c \rangle )' - 0.5*\langle Y, K^c \rangle  (\langle  K^c , K^c \rangle )') \\
	///    = 1/\langle K^c,K^c \rangle ^{3/2} \langle  \langle K^c,K^c \rangle  (Y -2 y u^T + my u u^T)- \langle Y, K^c \rangle (K -2 k u^T+ mk u u^T),K'  \rangle \\
	///    = 1/\langle K^c,K^c \rangle ^{3/2} \langle W,K' \rangle \f]
	///reordering rsults in
	/// \f[ W= \langle K^c,K^c \rangle  Y - \langle Y, K^c \rangle K \\
	///     - 2 (\langle K^c,K^c \rangle y - \langle Y, K^c \rangle k) u^T \\
	///     +   (\langle K^c,K^c \rangle my - \langle Y, K^c \rangle mk) u u^T \f]
	/// where \f$ K' \f$ is the derivative of K with respct of the kernel parameters.
	ResultType evalDerivative( const SearchPointType & input, FirstOrderDerivative & derivative ) const {
		mep_kernel->setParameterVector(input);
		// the drivative is calculated in two sweeps of the data. first the required statistics
		// \langle K^c,K^c \rangle , mk and k are evaluated exactly as in eval

		KernelMatrixResults results = evaluateKernelMatrix();

		std::size_t parameters = mep_kernel->numberOfParameters();
		derivative.resize(parameters);
		derivative.clear();
		SHARK_PARALLEL_FOR(int i = 0; i < (int)m_data.numberOfBatches(); ++i){
			std::size_t startX = 0;
			for(int j = 0; j != i; ++j){
				startX+= batchSize(m_data.batch(j));
			}
			RealVector threadDerivative(parameters,0.0);
			RealVector blockDerivative;
			boost::shared_ptr<State> state = mep_kernel->createState();
			RealMatrix blockK;//block of the KernelMatrix
			RealMatrix blockW;//block of the WeightMatrix
			std::size_t startY = 0;
			for(int j = 0; j <= i; ++j){
				mep_kernel->eval(m_data.batch(i).input,m_data.batch(j).input,blockK,*state);
				mep_kernel->weightedParameterDerivative(
					m_data.batch(i).input,m_data.batch(j).input,
					generateDerivativeWeightBlock(i,j,startX,startY,blockK,results),//takes symmetry into account
					*state,
					blockDerivative
				);
				noalias(threadDerivative) += blockDerivative;
				startY += batchSize(m_data.batch(j));
			}
			SHARK_CRITICAL_REGION{
				noalias(derivative) += threadDerivative;
			}
		}
		derivative *= -1;
		derivative /= m_elements;
		return -results.error;
	}

private:
	AbstractKernelFunction<InputType>* mep_kernel;     ///< kernel function
	LabeledData<InputType,LabelType> m_data;      ///< data points
	RealVector m_columnMeanY;                        ///< mean label vector
	double m_meanY;                                  ///< mean label element
	std::size_t m_numberOfClasses;                  ///< number of classes
	std::size_t m_elements;                          ///< number of data points
	bool m_centering;

	struct KernelMatrixResults{
		RealVector k;
		double KcKc;
		double YcKc;
		double error;
		double meanK;
	};

	void setupY(Data<unsigned int>const& labels, bool centering){
		//preprocess Y so calculate column means and overall mean
		//the most efficient way to do this is via the class counts
		std::vector<std::size_t> classCount = classSizes(labels);
		m_numberOfClasses = classCount.size();
		RealVector classMean(m_numberOfClasses);
		double dm1 = m_numberOfClasses-1.0;
		m_meanY = 0;
		for(std::size_t i = 0; i != m_numberOfClasses; ++i){
			classMean(i) = classCount[i]-(m_elements-classCount[i])/dm1;
			m_meanY += classCount[i] * classMean(i);
		}
		classMean /= m_elements;
		m_meanY /= sqr(double(m_elements));
		m_columnMeanY.resize(m_elements);
		std::size_t i = 0;
		for(auto const& batch:labels.batches()){
			std::size_t size = batch.size();
			for(std::size_t j = 0; j != batch.size(); ++j, ++i){
				m_columnMeanY(i) = classMean(batch(j));
			}
		}
		if(!centering){
			m_meanY = 0;
			m_columnMeanY.clear();
		}
	}

	void setupY(Data<RealVector>const& labels, bool centering){
		RealVector meanLabel = mean(labels);
		m_columnMeanY.resize(m_elements);
		std::size_t i = 0;
		for(auto const& batch:labels.batches()){
			std::size_t size = batch.size1();
			noalias(subrange(m_columnMeanY, i, i+ size)) = batch % meanLabel;
			i+= size;
		}
		m_meanY = inner_prod(meanLabel,meanLabel);
		if(!centering){
			m_meanY = 0;
			m_columnMeanY.clear();
		}
	}
	void computeBlockY(UIntVector const& labelsi,UIntVector const& labelsj, RealMatrix& blockY)const{
		std::size_t blockSize1 = labelsi.size();
		std::size_t blockSize2 = labelsj.size();
		double dm1 = m_numberOfClasses-1.0;
		for(std::size_t k = 0; k != blockSize1; ++k){
			for(std::size_t l = 0; l != blockSize2; ++l){
				if( labelsi(k) ==  labelsj(l))
					blockY(k,l) = 1;
				else
					blockY(k,l) = -1.0/dm1;
			}
		}
	}

	void computeBlockY(RealMatrix const& labelsi,RealMatrix const& labelsj, RealMatrix& blockY)const{
		noalias(blockY) = labelsi % trans(labelsj);
	}
    
    /// Update a sub-block of the matrix \f$ \langle Y, K^x \rangle \f$.
	double updateYK(UIntVector const& labelsi,UIntVector const& labelsj, RealMatrix const& block)const{
		std::size_t blockSize1 = labelsi.size();
		std::size_t blockSize2 = labelsj.size();
		//todo optimize the i=j case
		double result = 0;
		double dm1 = m_numberOfClasses-1.0;
		for(std::size_t k = 0; k != blockSize1; ++k){
			for(std::size_t l = 0; l != blockSize2; ++l){
				if(labelsi(k) == labelsj(l))
					result += block(k,l);
				else
					result -= block(k,l)/dm1;
			}
		}
		return result;
	}

	/// Update a sub-block of the matrix \f$ \langle Y, K^x \rangle \f$.
	double updateYK(RealMatrix const& labelsi,RealMatrix const& labelsj, RealMatrix const& block)const{
        RealMatrix Y(labelsi.size1(), labelsj.size1());
        computeBlockY(labelsi,labelsj,Y);
        return sum(Y * block);
	}

	/// Compute a sub-block of the matrix
	/// \f[ W = \langle K^c, K^c \rangle Y - \langle Y, K^c \rangle K -2 (\langle K^c, K^c \rangle y - \langle Y, K^c \rangle k) u^T + (\langle K^c, K^c \rangle my - \langle Y, K^c \rangle mk) u u^T \f]
	RealMatrix generateDerivativeWeightBlock(
		std::size_t i, std::size_t j,
		std::size_t start1, std::size_t start2,
		RealMatrix const& blockK,
		KernelMatrixResults const& matrixStatistics
	)const{
		std::size_t blockSize1 = batchSize(m_data.batch(i));
		std::size_t blockSize2 = batchSize(m_data.batch(j));
		//double n = m_elements;
		double KcKc = matrixStatistics.KcKc;
		double YcKc = matrixStatistics.YcKc;
		double meanK = matrixStatistics.meanK;
		RealMatrix blockW(blockSize1,blockSize2);

		//first calculate \langle Kc,Kc \rangle  Y.
		computeBlockY(m_data.batch(i).label,m_data.batch(j).label,blockW);
		blockW *= KcKc;
		//- \langle Y,K^c \rangle K
		blockW-=YcKc*blockK;
		//  -2(\langle Kc,Kc \rangle y -\langle Y, K^c \rangle  k) u^T
		// implmented as: -(\langle K^c,K^c \rangle y -2\langle Y, K^c \rangle  k) u^T -u^T(\langle K^c,K^c \rangle y -2\langle Y, K^c \rangle  k)^T
		//todo find out why this is correct and the calculation above is not.
		blockW-=repeat(subrange(KcKc*m_columnMeanY - YcKc*matrixStatistics.k,start2,start2+blockSize2),blockSize1);
		blockW-=trans(repeat(subrange(KcKc*m_columnMeanY - YcKc*matrixStatistics.k,start1,start1+blockSize1),blockSize2));
		// + (\langle Kc,Kc \rangle  my-2\langle Y, Kc \rangle mk) u u^T
		blockW+= KcKc*m_meanY-YcKc*meanK;
		blockW /= KcKc*std::sqrt(KcKc);
		//symmetry
		if(i != j)
			blockW *= 2.0;
		return blockW;
	}

	/// \brief Evaluate the centered kernel Gram matrix.
	///
	/// The computation is as follows. By means of a
	/// number of identities it holds
	/// \f[ \langle K^c, K^c \rangle = \\
	///     \langle K^c, K^c \rangle  = \langle K,K \rangle  -2n\langle k,k \rangle  +mk^2n^2 \\
	///     \langle K^c, Y \rangle  = \langle K, Y \rangle  - 2 n \langle k, y \rangle  + n^2 mk my \f]
	/// where k is the row mean over K and y the row mean over y, mk, my the total means of K and Y
	/// and n the number of elements
	KernelMatrixResults evaluateKernelMatrix()const{
		//it holds
		// \langle K^c,K^c \rangle  = \langle K,K \rangle  -2n\langle k,k \rangle  +mk^2n^2
		// \langle K^c,Y \rangle  = \langle K, Y \rangle  - 2 n \langle k, y \rangle  + n^2 mk my
		// where k is the row mean over K and y the row mean over y, mk, my the total means of K and Y
		// and n the number of elements

		double KK = 0; //stores \langle K,K \rangle
		double YK = 0; //stores \langle Y,K^c \rangle
		RealVector k(m_elements,0.0);//stores the row/column means of K
		SHARK_PARALLEL_FOR(int i = 0; i < (int)m_data.numberOfBatches(); ++i){
			std::size_t startRow = 0;
			for(int j = 0; j != i; ++j){
				startRow+= batchSize(m_data.batch(j));
			}
			std::size_t rowSize = batchSize(m_data.batch(i));
			double threadKK = 0;
			double threadYK = 0;
			RealVector threadk(m_elements,0.0);
			std::size_t startColumn = 0; //starting column of the current block
			for(int j = 0; j <= i; ++j){
				std::size_t columnSize = batchSize(m_data.batch(j));
				RealMatrix blockK = (*mep_kernel)(m_data.batch(i).input,m_data.batch(j).input);
				if(i == j){
					threadKK += frobenius_prod(blockK,blockK);
					subrange(threadk,startColumn,startColumn+columnSize)+=sum(as_columns(blockK));//update sum_rows(K)
					threadYK += updateYK(m_data.batch(i).label,m_data.batch(j).label,blockK);
				}
				else{//use symmetry ok K
					threadKK += 2.0 * frobenius_prod(blockK,blockK);
					subrange(threadk,startColumn,startColumn+columnSize)+=sum(as_columns(blockK));
					subrange(threadk,startRow,startRow+rowSize)+=sum(as_rows(blockK));//symmetry: block(j,i)
					threadYK += 2.0 * updateYK(m_data.batch(i).label,m_data.batch(j).label,blockK);
				}
				startColumn+=columnSize;
			}
			SHARK_CRITICAL_REGION{
				KK += threadKK;
				YK +=threadYK;
				noalias(k) +=threadk;
			}
		}
		//calculate the error
		double n = (double)m_elements;
		k /= n;//means
		double meanK = sum(k)/n;
		if(!m_centering){
			k.clear();
			meanK = 0;
		}
		double n2 = sqr(n);
		double YcKc = YK-2.0*n*inner_prod(k,m_columnMeanY)+n2*m_meanY*meanK;
		double KcKc = KK - 2.0*n*inner_prod(k,k)+n2*sqr(meanK);

		KernelMatrixResults results;
		results.k=k;
		results.YcKc = YcKc;
		results.KcKc = KcKc;
		results.meanK = meanK;
		results.error = YcKc/std::sqrt(KcKc)/n;
		return results;
	}
};


}
#endif
