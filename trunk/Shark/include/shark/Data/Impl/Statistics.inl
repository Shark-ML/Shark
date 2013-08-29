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
template<class Vec1T,class Vec2T,class Vec3T>
void meanvar
(
	Data<Vec1T> const& data,
	blas::vector_container<Vec2T>& meanVec,
	blas::vector_container<Vec3T>& varianceVec
)
{
	SIZE_CHECK(!data.empty());
	typedef typename Data<Vec1T>::const_batch_reference BatchRef;
	std::size_t const dataSize = data.numberOfElements();
	std::size_t elementSize=dataDimension(data);

	varianceVec().resize(elementSize);
	varianceVec().clear();
	
	meanVec()= mean(data);
	
	//sum of variances of each column
	BOOST_FOREACH(BatchRef batch,data.batches()){
		std::size_t batchSize = batch.size1();
		noalias(varianceVec()) += sumRows(sqr(batch-repeat(meanVec,batchSize)));
	}
	varianceVec()/=dataSize;
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
template<class Vec1T,class Vec2T,class MatT>
void meanvar
(
	const Data<Vec1T>& data,
	blas::vector_container<Vec2T>& meanVec,
	blas::matrix_container<MatT>& covariance
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
		symmRankKUpdate(trans(batch),covariance,1.0);
	}
	covariance()/=dataSize;
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
	
	typedef typename Data<VectorType>::const_batch_reference BatchRef; 
	 
	BOOST_FOREACH(BatchRef batch, data.batches()){
		mean += sumRows(batch);
	}
	mean /= data.numberOfElements();
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
VectorType variance(const Data<VectorType>& data)
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
typename VectorMatrixTraits<VectorType>::DenseMatrixType covariance(const Data<VectorType>& data) {
	RealVector mean;
	RealMatrix covariance;
	meanvar(data,mean,covariance);
	return covariance;
}

}
