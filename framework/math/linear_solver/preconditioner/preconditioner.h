#pragma once
#include "framework/math/vector.h"

namespace pdes
{
  template<typename VectorType = Vector<>>
  class Preconditioner
  {
  public:
    Preconditioner() = default;
    virtual ~Preconditioner() = default;

    /// Apply the preconditioner: dst = inv * src
    virtual void vmult(const VectorType& src, VectorType& dst) const = 0;

    /// Optional: name for logging/debugging
    virtual std::string name() const = 0;
  };
}
