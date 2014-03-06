/*!
 *
 *
 * \brief       Implements \f$(\mu,\lambda)\f$ selection.
 *
 *
 *
 * \author      T.Voss
 * \date        2010-2011
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
#ifndef SHARK_EA_MU_KOMMA_LAMBDA_SELECTION_H
#define SHARK_EA_MU_KOMMA_LAMBDA_SELECTION_H

#include <shark/Core/Exception.h>

#include <algorithm>

namespace shark {

namespace tag {
/**
 * \brief Tags the selection scheme.
 */
struct MuKommaLambda {};
}

/**
 * \brief Selects lambda offspring individuals from mu parents
 * \throws shark::Exception if std::distance( beginParents,endParents ) >= std::distance( beginOffspring, endOffspring ).
 */
template<typename ParentsIterator, typename OffspringIterator>
void select_mu_komma_lambda(ParentsIterator beginParents,
                            ParentsIterator endParents,
                            OffspringIterator beginOffspring,
                            OffspringIterator endOffspring
                           )
{
	std::size_t mu = std::distance(beginParents, endParents);
	std::size_t lambda = std::distance(beginOffspring, endOffspring);

	if (mu >= lambda)
		throw(shark::Exception("Lambda needs to be larger than mu.", __FILE__, __LINE__));

	std::sort(beginOffspring, endOffspring);
	std::copy(beginOffspring, beginOffspring + mu, beginParents);
}

/**
 * \brief Selects lambda offspring individuals from mu parents. Relies on the supplied predicate for comparing individuals.
 * \throws shark::Exception if std::distance( beginParents,endParents ) >= std::distance( beginOffspring, endOffspring ).
 */
template<typename ParentsIterator, typename OffspringIterator, typename Relation>
void select_mu_komma_lambda_p(ParentsIterator beginParents,
                              ParentsIterator endParents,
                              OffspringIterator beginOffspring,
                              OffspringIterator endOffspring,
                              Relation relation
                             )
{
	std::size_t mu = std::distance(beginParents, endParents);
	std::size_t lambda = std::distance(beginOffspring, endOffspring);

	if (mu >= lambda)
		throw(shark::Exception("Lambda needs to be larger than mu.", __FILE__, __LINE__));

	std::sort(beginOffspring, endOffspring, relation);
	std::copy(beginOffspring, beginOffspring + mu, beginParents);
}



}

#endif
