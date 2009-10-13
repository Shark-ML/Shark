//===========================================================================
/*!
 *  \file statistics.h
 *
 *  \brief functions for basic statistics
 *
 *  \author  C. Igel
 *
 *  \par Copyright (c) 1998-2008:
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
//===========================================================================

#ifndef STATISTICS_H
#define STATISTICS_H

#include<algorithm>
#include<SharkDefs.h>
#include <vector>
namespace Shark {

/*! 
 * \brief compute median
 */
template<class T>
double  median(std::vector<T> &v) {
  if(v.empty()) throw SHARKEXCEPTION("[median] list must not be empty");
  sort(v.begin(),v.end());
  if(v.size() % 2) return v[(v.size() - 1) / 2];
  return double(v[v.size() / 2] + v[v.size()  / 2 - 1]) / 2.;
}

/*! 
 * \brief compute percentilee (Excel way) 
 */
template<class T>
double  percentile(std::vector<T> &v, double p =.25) {
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
double  nth(std::vector<T> &v, unsigned n) {
  if(v.empty()) throw SHARKEXCEPTION("[nth] list must not be empty");
  if(v.size() <= n) throw SHARKEXCEPTION("[nth] n must not be larger than number of itmes in list");
  sort(v.begin(),v.end());
  return v[n];
}

/*! 
 * \brief compute mean 
 */
template<class T>
double  mean(std::vector<T> v) {
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
double correlation(std::vector<T> &v1, std::vector<T> &v2)
{
  double square_1  = 0;
  double sum_1     = 0;
  double square_2  = 0;
  double sum_2     = 0;
  double square_12 = 0;
  double sum_12    = 0;

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
  double cor = (cov / sqrt( var_1 * var_2 ));

  cov   /= (n-1);
  var_1 /= (n-1);
  var_2 /= (n-1);

  return cor;
}

/*! 
 * \brief compute sample variance  
 */
template<class T>
  double variance(std::vector<T> &v, bool unbiased=true)
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


}
#endif 
