//===========================================================================
/*!
 *  \brief The k-means clustering algorithm.
 *
 *  \author T. Glasmachers
 *  \date 2011
 *
 *  \par Copyright (c) 1998-2011:
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


#ifndef SHARK_ALGORITHMS_KMEANS_H
#define SHARK_ALGORITHMS_KMEANS_H


#include <shark/Data/Dataset.h>
#include <shark/Models/Clustering/Centroids.h>


namespace shark{


///
/// \brief The k-means clustering algorithm.
///
/// \par
/// The k-means algorithm takes vector-valued data
/// \f$ \{x_1, \dots, x_\ell\} \subset \mathbb R^d \f$
/// and splits it into k clusters, based on centroids
/// \f$ \{c_1, \dots, c_k\} \f$.
/// The result is stored in a Centroids object that can be used to
/// construct clustering models.
///
/// \par
/// This implementation starts the search with the given centroids,
/// in case the provided centroids object (third parameter) contains
/// a set of k centroids. Otherwise the search starts from the first
/// k data points.
///
/// \par
/// Note that the data set needs to include at least k data points
/// for k-means to work. This is because the current implementation
/// does not allow for empty clusters.
///
/// \param data           vector-valued data to be clustered
/// \param k              number of clusters
/// \param centroids      centroids input/output
/// \param maxIterations  maximum number of k-means iterations; 0: unlimited
/// \return               number of k-means iterations
///

std::size_t kMeans(Data<RealVector> const& data, std::size_t k, Centroids& centroids, std::size_t maxIterations = 0);


}
#endif
