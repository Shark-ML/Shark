/**
 *
 *  \brief Traits which allow to define ProxyReferences for types
 *
 * A ProxyReference can be used in the context of abstract functions to bind several related types
 * of arguments to a single proxy type. Main use are ublas expression templates so that
 * vectors, matrix rows and subvectors can be treated as one argument type
 *
 *  \author O.Krause
 *  \date 2012
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
#ifndef SHARC_CORE_PROXYREFERENCETRAITS_H
#define SHARC_CORE_PROXYREFERENCETRAITS_H
#include <string>
#include <shark/LinAlg/Base.h>
namespace shark {
	///\brief sets the type of ProxxyReference
	///
	///A Proxy referencecan be used when several related types are to be treated by one proxy type
	///argument, for example expression templates
	template<class T>
	struct ConstProxyReference{
		typedef T const& type;
	};

	template<class T>
	struct ConstProxyReference<blas::vector<T> >{
		typedef blas::dense_vector_adaptor<T const> const& type;
	};
	template<class T>
	struct ConstProxyReference<blas::vector<T> const>{
		typedef blas::dense_vector_adaptor<T const> const& type;
	};
	template<class T>
	struct ConstProxyReference<blas::compressed_vector<T> >{
		typedef blas::sparse_vector_adaptor<T const,std::size_t> const& type;
	};
		template<class T>
	struct ConstProxyReference<blas::compressed_vector<T> const >{
		typedef blas::sparse_vector_adaptor<T const,std::size_t> const& type;
	};
	template<class T>
	struct ConstProxyReference<blas::matrix<T> >{
		typedef blas::dense_matrix_adaptor<T const,blas::row_major> const& type;
	};
	template<class T>
	struct ConstProxyReference<blas::matrix<T> const >{
		typedef blas::dense_matrix_adaptor<T const,blas::row_major> const& type;
	};
}
#endif
