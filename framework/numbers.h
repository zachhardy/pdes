#pragma once
#include "framework/types.h"
#include <limits>

namespace pdes
{
  namespace numbers
  {
    constexpr unsigned int invalid_unsigned_int = static_cast<unsigned int>(-1);
    constexpr types::global_index invalid_global_index = static_cast<types::global_index>(-1);
  }
}
