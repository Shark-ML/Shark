//===========================================================================
/*!
*  \brief Test case for k-means clustering.
*
*  \author  T. Glasmachers
*  \date    2011
*
*  \par Copyright (c) 2011:
*      Institut f&uuml;r Neuroinformatik<BR>
*      Ruhr-Universit&auml;t Bochum<BR>
*      D-44780 Bochum, Germany<BR>
*      Phone: +49-234-32-27974<BR>
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

#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#define BOOST_TEST_MODULE Algorithms_KMeans
#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <algorithm>

#include <shark/Algorithms/KMeans.h>


using namespace shark;


BOOST_AUTO_TEST_CASE(KMeans)
{
	RealVector v(1);

	// prepare data set
	std::vector<RealVector> data(300);
	for (std::size_t i=0; i<100; i++)
	{
		v(0) = Rng::uni();
		data[i] = v;
		v(0) = Rng::uni() + 10.0;
		data[100+i] = v;
		v(0) = Rng::uni() + 20.0;
		data[200+i] = v;
	}
	Data<RealVector> dataset = createDataFromRange(data);

	// prepare initial centroids
	std::vector<RealVector> start(3);
	v(0) =  2.0; start[0] = v;
	v(0) =  7.0; start[1] = v;
	v(0) = 25.0; start[2] = v;
	Centroids centroids( createDataFromRange(start));

	// invoke k-means
	std::size_t iterations = kMeans(dataset, 3, centroids);
	std::cout<<iterations<<std::endl;

	// check result
	Data<RealVector> const& c = centroids.centroids();
	std::cout<<c<<std::endl;
	BOOST_CHECK_EQUAL(c.numberOfElements(), 3u);
	BOOST_CHECK(c.element(0)(0) >  0.0);
	BOOST_CHECK(c.element(0)(0) <  1.0);
	BOOST_CHECK(c.element(1)(0) > 10.0);
	BOOST_CHECK(c.element(1)(0) < 11.0);
	BOOST_CHECK(c.element(2)(0) > 20.0);
	BOOST_CHECK(c.element(2)(0) < 21.0);
	BOOST_CHECK_LE(iterations, 3u);
}
