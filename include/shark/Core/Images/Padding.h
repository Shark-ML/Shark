#ifndef SHARK_CORE_IMAGE_PADDING_H
#define SHARK_CORE_IMAGE_PADDING_H

#include <shark/LinAlg/Base.h>
namespace shark{
enum class Padding{
	Valid,
	ZeroPad,
	RepeatBorder
};

}

#endif