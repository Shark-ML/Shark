//===========================================================================
/*!
 *  \file VectorStatistics.h
 *
 *  some functions for vector valued statistics like mean, variance and covariance
 *
 *  \author O.Krause, C. Igel
 *  \date 2010-2011
 *
 *  \par Copyright (c) 1998-2007:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
 *      Fax:   +49-234-32-14209<BR>
 *      eMail: Shark-admin@neuroinformatik.ruhr-uni-bochum.de<BR>
 *      www:   http://www.neuroinformatik.ruhr-uni-bochum.de<BR>
 *      <BR>
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
#ifndef SHARK_LINALG_VECTORSTATISTICS_H
#define SHARK_LINALG_VECTORSTATISTICS_H

#include <shark/LinAlg/Base.h>
#include <shark/Data/Dataset.h>

#include <algorithm>
#include <vector>

namespace shark{
	
/**
* \ingroup shark_globals
* 
* @{
*/

//! Calculates the mean and variance values of the input data
template<class Vec1T,class Vec2T,class Vec3T>
void meanvar
(
	const Data<Vec1T>& data,
	blas::vector_container<Vec2T>& mean,
	blas::vector_container<Vec3T>& variance
);
//! Calculates the mean, variance and covariance values of the input data
template<class Vec1T,class Vec2T,class MatT>
void meanvar
(
	const Data<Vec1T>& data,
	blas::vector_container<Vec2T>& mean,
	blas::matrix_container<MatT>& variance
);
//! Calculates the mean, variance and covariance values of the input data
template<class MatT, class Vec1T,class Vec2T>
void meanvar
(
	blas::matrix_container<MatT>& data,
	blas::vector_container<Vec1T>& meanVec,
	blas::vector_container<Vec2T>& varianceVec
);

//! Calculates the coefficient of correlation matrix of the data vectors
template<class VectorType>
typename VectorMatrixTraits<VectorType>::DenseMatrixType corrcoef(Data<VectorType> const& data);

//! Calculates the mean vector of the input vectors.
template<class VectorType>
VectorType mean(Data<VectorType> const& data);

template<class VectorType>
VectorType mean(UnlabeledData<VectorType> const& data){
	return mean(static_cast<Data<VectorType> const&>(data));
}

//! Calculates the mean vector of the input vectors.
template<class MatrixType>
blas::vector<typename MatrixType::value_type>
mean(const blas::matrix_container<MatrixType>& data);

//this fails...or the above, but we need both...hmm
//~ //! Calculates the mean of the values stored in a general range
//~ template<class Range>
//~ typename boost::range_value<Range>::type mean(Range const& data){
	//~ typedef typename boost::range_iterator<Range const>::type Iterator;
	//~ Iterator pos = boost::begin(data);
	//~ Iterator end = boost::end(data);
	//~ typename boost::range_value<Range>::type m = *pos;
	//~ ++pos;
	//~ for(; pos != end; ++pos)
		//~ m+=*pos;
	//~ m /= boost::size(data);
	//~ return m;
//~ }

//! Calculates the variance vector of the input vectors
template<class VectorType>
VectorType variance(const Data<VectorType>& data);

//! Calculates the covariance matrix of the data vectors
template<class VectorType>
typename VectorMatrixTraits<VectorType>::DenseMatrixType covariance(const Data<VectorType>& data);


/*! 
 * \brief compute median
 */
template<class T>
double stl_median(std::vector<T> &v) {
	if(v.empty()) throw SHARKEXCEPTION("[median] list must not be empty");
	sort(v.begin(),v.end());
	if(v.size() % 2) return v[(v.size() - 1) / 2];
	return double(v[v.size() / 2] + v[v.size()  / 2 - 1]) / 2.;
}

/*! 
 * \brief compute percentilee (Excel way) 
 */
template<class T>
double stl_percentile(std::vector<T> &v, double p =.25) {
	if(v.empty()) throw SHARKEXCEPTION("[percentile] list must not be empty");;
	sort(v.begin(),v.end());
	unsigned N = v.size();
	
	double n = p * (double(N) - 1.) + 1.;
	unsigned k = unsigned(floor(n));
	double d = n - k;
	
	if(k == 0) return v[0];
	if(k == N) return v[N - 1];
	
	return v[k - 1] + d * (v[k] - v[k - 1]);
}

/*! 
 * \brief return nth element after sorting
 */
template<class T>
double nth(std::vector<T> &v, unsigned n) {
	if(v.empty()) throw SHARKEXCEPTION("[nth] list must not be empty");
	if(v.size() <= n) throw SHARKEXCEPTION("[nth] n must not be larger than number of itmes in list");
	sort(v.begin(),v.end());
	return v[n];
}

/*! 
 * \brief compute mean 
 */
template<class T>
double stl_mean(std::vector<T> v) {
	double sum = 0;
	if(v.empty()) throw  SHARKEXCEPTION("[mean] list must not be empty");
	unsigned n = v.size();
	for(unsigned i=0; i<n; i++) sum += v[i];
	return sum/n;
}
	
/*! 
 * \brief compute variance 
 */
template<class T>
double stl_correlation(std::vector<T> &v1, std::vector<T> &v2)
{
	double square_1  = 0;
	double sum_1     = 0;
	double square_2  = 0;
	double sum_2     = 0;
	double square_12 = 0;

	if(v1.empty()) throw  SHARKEXCEPTION("[correlation] list must not be empty");
	if(v1.size() != v2.size()) throw SHARKEXCEPTION("[correlation] samples must have same size");
	
	unsigned n = v1.size();
	
	for(unsigned i=0; i<n; i++) {
		square_1 += v1[i] * v1[i];
		sum_1 += v1[i];
		square_2 += v2[i] * v2[i];
		sum_2 += v2[i];
		square_12 += v1[i] * v2[i];
	}
	double average_1 = sum_1 / n;
	double average_2 = sum_2 / n;
	
	double var_1 = square_1 - n * average_1 * average_1;
	double var_2 = square_2 - n * average_2 * average_2;
	
	double cov = square_12 - n * average_1 * average_2;
	double cor = (cov / std::sqrt( var_1 * var_2 ));
	
	cov   /= (n-1);
	var_1 /= (n-1);
	var_2 /= (n-1);
	
	return cor;
}
	
/*! 
 * \brief compute sample variance  
 */
template<class T>
double stl_variance(std::vector<T> &v, bool unbiased=true)
{
	double square = 0;
	double sum = 0;
	if(v.empty())  SHARKEXCEPTION("[variance] list must not be empty");
	unsigned n = v.size();
	for(unsigned i=0; i<n; i++) {
		square += v[i] * v[i];
		sum += v[i];
	}
	if (unbiased)
		return (square - sum * sum / n) / (n - 1);
	else 
		return (square - sum * sum / n) / n;
}
	
	
/** @}*/
}

#include "Impl/VectorStatistics.inl"

#endif // SHARK_LINALG_VECTORSTATISTICS_H
