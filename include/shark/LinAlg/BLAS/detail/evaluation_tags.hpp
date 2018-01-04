//===========================================================================
/*!
 * 
 *
 * \brief       Tags representing different type of expression evaluation categories
 *
 * \author      O. Krause
 * \date        2016
 *
 *
 * \par Copyright 1995-2015 Shark Development Team
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
//===========================================================================

#ifndef REMORA_DETAIL_EVALUATION_TAGS_HPP
#define REMORA_DETAIL_EVALUATION_TAGS_HPP

namespace remora{
	
// Evaluation type tags:
// dense_tag -> dense storage scheme an dense interface supported
// continuous_dense_tag -> dense storage scheme where stride between all elements is 1
// sparse_tag -> sparse storage scheme and supports sparse interface.
// packed_tag ->BLAS packed format and supports packed interface
// unknown_tag -> no known storage scheme, only supports basic interface(probably blockwise evaluation)
struct unknown_tag{};
struct sparse_tag:public unknown_tag{};
struct dense_tag: public unknown_tag{};
struct continuous_dense_tag: public dense_tag{};
struct packed_tag: public unknown_tag{};
	
struct elementwise_tag{};
struct blockwise_tag{};

//evaluation categories
template<class Tag>
struct elementwise: public elementwise_tag{
	typedef Tag tag;
};
template<class Tag>
struct blockwise: public blockwise_tag{
	typedef Tag tag;
};


template<class Tag1, class Tag2>
struct evaluation_tag_restrict_traits{
	typedef Tag1 type;
};

template<class Tag1>
struct evaluation_tag_restrict_traits<Tag1, dense_tag> {
	typedef dense_tag type;
};

template<>
struct evaluation_tag_restrict_traits<packed_tag,sparse_tag> {
	typedef sparse_tag type;
};

namespace detail{
	template<class Category1, class Category2>
	struct evaluation_restrict_traits{
		typedef blockwise<typename evaluation_tag_restrict_traits<
			typename Category1::tag, typename Category2::tag
		>::type> type;
	};
	template<class Tag1, class Tag2>
	struct evaluation_restrict_traits<elementwise<Tag1>, elementwise<Tag2> >{
		typedef elementwise<typename evaluation_tag_restrict_traits<Tag1, Tag2>::type> type;
	};
}
template<class E1, class E2>
struct evaluation_restrict_traits: public detail::evaluation_restrict_traits<
	typename E1::evaluation_category,
	typename E2::evaluation_category
>{};

}

#endif
