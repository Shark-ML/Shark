/**
 *
 *  \brief Implements \f$(\mu,\lambda)\f$ selection.
 *
 *  \author T.Voss
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
) {
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
) {
	std::size_t mu = std::distance(beginParents, endParents);
	std::size_t lambda = std::distance(beginOffspring, endOffspring);

	if (mu >= lambda)
		throw(shark::Exception("Lambda needs to be larger than mu.", __FILE__, __LINE__));

	std::sort(beginOffspring, endOffspring, relation);
	std::copy(beginOffspring, beginOffspring + mu, beginParents);
}



}

#endif
