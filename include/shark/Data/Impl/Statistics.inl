/*!
 * \brief       Implements the statistics module of shark datasets
 * 
 * \author      O. Krause
 * \date        2015
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
namespace shark{ 
	
/*!
 *  \brief Calculates the mean and variance values of a dataset
 *
 *  Given the vector of data, the mean and variance values
 *  are calculated as in the functions #mean and #variance.
 *
 *      \param  data Input data.
 *      \param  meanVec Vector of mean values.
 *      \param  varianceVec Vector of variances.
 *
 */
template<class Vec1T,class Vec2T,class Vec3T, class Device>
void meanvar
(
	Data<Vec1T> const& data,
	blas::vector_container<Vec2T, Device>& meanVec,
	blas::vector_container<Vec3T, Device>& varianceVec
)
{
	SIZE_CHECK(!data.empty());
	std::size_t const dataSize = data.numberOfElements();
	std::size_t elementSize=dataDimension(data);

	varianceVec().resize(elementSize);
	varianceVec().clear();
	
	meanVec()= mean(data);
	
	//sum of variances of each column
	for(auto& batch: data.batches()){
		std::size_t batchSize = batch.size1();
		noalias(varianceVec()) += norm_sqr(as_columns(batch-repeat(meanVec,batchSize)));
	}
	varianceVec() /= double(dataSize);
}

/*!
 *  \brief Calculates the mean and covariance values of a set of data
 *
 *  Given the vector of data, the mean and variance values
 *  are calculated as in the functions #mean and #variance.
 *
 *      \param  data Input data.
 *      \param  meanVec Vector of mean values.
 *      \param  covariance Covariance matrix.
 *
 */
template<class Vec1T,class Vec2T,class MatT, class Device>
void meanvar
(
	Data<Vec1T> const& data,
	blas::vector_container<Vec2T, Device>& meanVec,
	blas::matrix_container<MatT, Device>& covariance
){
	SIZE_CHECK(!data.empty());
	typedef typename Batch<Vec1T>::type BatchType;
	std::size_t const dataSize = data.numberOfElements();
	std::size_t elementSize=dataDimension(data);

	covariance().resize(elementSize,elementSize);
	covariance().clear();
	
	meanVec() = mean(data);
	//sum of variances of each column
	for(std::size_t b = 0; b != data.numberOfBatches(); ++b){
		//make the batch mean-free
		BatchType batch = data.batch(b)-repeat(meanVec,data.batch(b).size1());
		noalias(covariance) += prod(trans(batch),batch);
	}
	covariance() /= double(dataSize);
}

/*!
 *  \brief Calculates the mean vector of array "x".
 *
 *  Given a \em d -dimensional array \em x with size \em N1 x ... x \em Nd,
 *  this function calculates the mean vector given as:
 *  \f[
 *      mean_j = \frac{1}{N1} \sum_{i=1}^{N1} x_{i,j}
 *  \f]
 *  Example:
 *  \f[
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1 &  2 &  3 &  4\\
 *          5 &  6 &  7 &  8\\
 *          9 & 10 & 11 & 12\\
 *      \end{array}
 *      \right)
 *      \longrightarrow
 *      \frac{1}{3}
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          1+5+9 & 2+6+10 & 3+7+11 & 4+8+12\\
 *      \end{array}
 *      \right)
 *      \longrightarrow
 *      \left(
 *      \begin{array}{*{4}{c}}
 *          5 &  6 &  7 &  8\\
 *      \end{array}
 *      \right)
 *  \f]
 *
 *      \param  data input data, from which the
 *                mean value will be calculated
 *      \return the mean vector of \em x
 */
template<class VectorType>
VectorType mean(Data<VectorType> const& data){
	SIZE_CHECK(!data.empty());

	VectorType mean(dataDimension(data),0.0);
	 
	for(auto& batch: data.batches()){
		mean += sum(as_columns(batch));
	}
	mean /= double(data.numberOfElements());
	return mean;
}

/*!
 *  \brief Calculates the variance vector of array "x".
 *
 *  Given a \em d -dimensional array \em x with size \em N1 x ... x \em Nd
 *  and mean value vector \em m,
 *  this function calculates the variance vector given as:
 *  \f[
 *      variance = \frac{1}{N1} \sum_{i=1}^{N1} (x_i - m_i)^2
 *  \f]
 *
 *      \param  data input data from which the variance will be calculated
 *      \return the variance vector of \em x
 */
template<class VectorType>
VectorType variance(Data<VectorType> const& data)
{
	RealVector m;   // vector of mean values.
	RealVector v;   // vector of variance values

	meanvar(data,m,v);
	return v;
}

/*!
 *  \brief Calculates the covariance matrix of the data vectors stored in
 *         data.
 *
 *  Given a Set \f$X = (x_{ij})\f$ of \f$n\f$ vectors with length \f$N\f$,
 *  the function calculates the covariance matrix given as
 *
 *  \f$
 *      Cov = (c_{kl}) \mbox{,\ } c_{kl} = \frac{1}{n - 1} \sum_{i=1}^n
 *      (x_{ik} - \overline{x_k})(x_{il} - \overline{x_l})\mbox{,\ }
 *      k,l = 1, \dots, N
 *  \f$
 *
 *  where \f$\overline{x_j} = \frac{1}{n} \sum_{i = 1}^n x_{ij}\f$ is the
 *  mean value of \f$x_j \mbox{,\ }j = 1, \dots, N\f$.
 *
 *  \param data The \f$n \times N\f$ input matrix.
 *  \return \f$N \times N\f$ matrix of covariance values.
 */
template<class VectorType>
blas::matrix<typename VectorType::value_type> covariance(Data<VectorType> const& data) {
	RealVector mean;
	RealMatrix covariance;
	meanvar(data,mean,covariance);
	return covariance;
}

}
