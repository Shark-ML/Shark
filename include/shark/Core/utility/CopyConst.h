/*!
 *  \par Copyright (c) 2010-2011:
 *      Institut f&uuml;r Neuroinformatik<BR>
 *      Ruhr-Universit&auml;t Bochum<BR>
 *      D-44780 Bochum, Germany<BR>
 *      Phone: +49-234-32-25558<BR>
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
#ifndef SHARK_DATA_CORE_COPYCONST_H
#define SHARK_DATA_CORE_COPYCONST_H

#include <boost/type_traits/remove_const.hpp>

namespace shark{

///\brief If U is a const Type, than T is also made const. 
template<class T, class U>
struct CopyConst{
	typedef typename boost::remove_const<T>::type type;
};
template<class T, class U>
struct CopyConst<T,U const>{
	typedef typename boost::remove_const<T>::type const type;
};


}

#endif

