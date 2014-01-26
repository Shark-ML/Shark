/*!
 * 
 *
 * \brief       Traits wich specifies whether a type is part of a vector space.
 * 
 * A ProxyReference can be used in the context of abstract functions to bind several related types
 * of arguments to a single proxy type. Main use are ublas expression templates so that
 * vectors, matrix rows and subvectors can be treated as one argument type
 * 
 * 
 *
 * \author      O.Krause
 * \date        2012
 *
 *
 * \par Copyright 1995-2014 Shark Development Team
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
#ifndef SHARK_CORE_ISVECTOR_H
#define SHARK_CORE_ISVECTOR_H
#include <shark/LinAlg/Base.h>
#include <boost/mpl/bool.hpp>
namespace shark {
	///\brief IsVector is a traits class evaluating to true if T implements the rquirements of a vector.
	///
	///If this is true for a given T, it is assumed that it supports the operations
	///(let t,u be objects of type T and s a floating point value)
	///s*u -> returns T
	///t+=u
	///also t must be default constructable, copy constructable and assignable.
	///typical examples are the linear algebra vectors and matrices.
	///this traits must be specialized to very type which is a vector since by default it is assumed for safety, that
	///a unknown type is not confirming to this requirements.
	template<class T>
	struct IsVector:public boost::mpl::false_{};

	template<class T>
	struct IsVector<blas::vector<T> >:public boost::mpl::true_{};
	template<class T>
	struct IsVector<blas::compressed_vector<T> >:public boost::mpl::true_{};
	template<class T>
	struct IsVector<blas::matrix<T> >:public boost::mpl::true_{};
	template<class T>
	struct IsVector<blas::compressed_matrix<T> >:public boost::mpl::true_{};
}
#endif
