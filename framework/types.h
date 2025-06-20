#pragma once
#include <cstdint>

namespace pdes
{
    namespace types
    {
#ifdef PDES_USE_SINGLE
    using real = float;
#else
        using real = double;
#endif

#ifdef PDES_USE_64BIT
    using global_index = uint64_t;
        using signed_global_index = int64_t;
#else
        using global_index = uint32_t;
        using signed_global_index = int32_t;
#endif
    }
}
