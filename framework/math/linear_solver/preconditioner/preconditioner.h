#pragma once
#include "framework/math/vector.h"

namespace pdes
{
  template<typename MatrixType>
  class Preconditioner
  {
  public:
    using value_type = typename MatrixType::value_type;
    using VectorType = typename MatrixType::VectorType;

    Preconditioner() = default;
    virtual ~Preconditioner() = default;

    /// Sets up the preconditioner for a given matrix.
    virtual void build(const MatrixType*) = 0;

    /// Apply the preconditioner: dst = inv * src
    virtual void vmult(const VectorType& src, VectorType& dst) const = 0;

    /// Optional: name for logging/debugging
    virtual std::string name() const = 0;
  };
}
