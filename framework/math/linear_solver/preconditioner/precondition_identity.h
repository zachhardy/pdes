#pragma once
#include "framework/math/linear_solver/preconditioner/preconditioner.h"
#include "framework/math/matrix.h"

namespace pdes
{
  /**
   * Identity preconditioner.
   *
   * This is a no-op preconditioner that simply copies the input vector to the output.
   * It is used as a default when no preconditioning is desired.
   */
  template<typename MatrixType = Matrix<>>
  class PreconditionIdentity final : public Preconditioner<MatrixType>
  {
  public:
    using value_type = typename Preconditioner<MatrixType>::value_type;
    using VectorType = typename Preconditioner<MatrixType>::VectorType;

    void build(const MatrixType*) override {}

    PreconditionIdentity() = default;
    explicit PreconditionIdentity(const MatrixType*) {}

    /// Applies z = r with no modification.
    void vmult(const VectorType& src, VectorType& dst) const override { dst = src; }

    /// Returns the name of the preconditioner.
    std::string name() const override { return "PreconditionIdentity"; }
  };
}
