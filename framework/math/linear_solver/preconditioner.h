#pragma once
#include "framework/math/vector.h"

namespace pdes
{
  template<typename Number = types::real>
  class Preconditioner
  {
  public:
    virtual ~Preconditioner() = default;

    /// Applies the preconditioner to a vector.
    virtual void vmult(const Vector<Number>& src,
                       Vector<Number>& dst) const = 0;

    /// Returns the name of the preconditioner.
    virtual std::string name() const = 0;
  };
}

