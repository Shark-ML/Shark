//===========================================================================
/*!
*
*  \brief Interface for nearest Neighbor queries
*
*  \author  O.Krause
*  \date    2012
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

#ifndef SHARK_ALGORITHMS_NEARESTNEIGHBORS_ABSTRACTNEARESTNEIGHBORS_H
#define SHARK_ALGORITHMS_NEARESTNEIGHBORS_ABSTRACTNEARESTNEIGHBORS_H

#include <shark/Core/utility/KeyValuePair.h>
#include <shark/Data/Dataset.h>
namespace shark{

	
///\brief Interface for Narest Neighbor queries.
///
///Defines the abstract interface for query of nearest neighbors. This is used to generalize over the different algorithms to query for
///nearest neighbors.
template<class InputType,class LabelType>
class AbstractNearestNeighbors{
public:
	typedef KeyValuePair<double,LabelType> DistancePair;
	typedef typename Batch<InputType>::type BatchInputType;

	///\brief Returns the k-nearest neighbors of a batch of points and returns them as linearized array.
	///
	///Given a batch of size n, a array with nxk values is returned where each ntry is a key-value pair of distance and label.
	///the first k entries are the neighbors of point 1, the next k of point 2 and so on.
	virtual std::vector<DistancePair> getNeighbors(BatchInputType const& batch, std::size_t k) const = 0;
	
	///\brief returns a const rference to the dataset used by the algorithm
	virtual LabeledData<InputType,LabelType>const& dataset()const = 0;
};

	
}

#endif