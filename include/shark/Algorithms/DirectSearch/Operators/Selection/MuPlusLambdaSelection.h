#ifndef SHARK_EA_MU_PLUS_LAMBDA_SELECTION_H
#define SHARK_EA_MU_PLUS_LAMBDA_SELECTION_H

#include <shark/Core/Exception.h>

#include <algorithm>

namespace shark {

namespace tag {
struct MuPlusLambda {};
}

template<
    typename ParentsIterator,
    typename OffspringIterator>
void select_mu_plus_lambda( ParentsIterator beginParents,
                            ParentsIterator endParents,
                            OffspringIterator beginOffspring,
                            OffspringIterator endOffspring
                          )
{
	std::sort( beginParents, endParents );
	std::sort( beginOffspring, endOffspring );
	std::merge( beginParents, endParents, beginOffspring, endOffspring, beginParents );
}

template<
    typename ParentsIterator,
    typename OffspringIterator,
    typename Relation>
void select_mu_plus_lambda_p( ParentsIterator beginParents,
                              ParentsIterator endParents,
                              OffspringIterator beginOffspring,
                              OffspringIterator endOffspring,
                              Relation relation
                            )
{
	std::sort( beginParents, endParents, relation );
	std::sort( beginOffspring, endOffspring, relation );
	std::merge( beginParents, endParents, beginOffspring, endOffspring, beginParents );
}
}

#endif
