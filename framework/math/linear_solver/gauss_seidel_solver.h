#pragma once
#include "framework/math/linear_solver/sor_solver.h"


namespace pdes
{
  /**
   * Gauss-Seidel iterative solver for Ax = b.
   * Templated on scalar type (default = types::real).
   */
  template<typename VectorType = Vector<>>
  class GaussSeidelSolver final : public SORSolver<VectorType>
  {
  public:
    using value_type = typename VectorType::value_type;
    using Result = typename LinearSolver<GaussSeidelSolver, VectorType>::Result;

    GaussSeidelSolver() = default;

    explicit GaussSeidelSolver(SolverControl* control)
      : SORSolver<VectorType>(control, value_type(1))
    {}

    std::string name() const override { return "GaussSeidelSolver"; }
  };
}
