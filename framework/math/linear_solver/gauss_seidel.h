#pragma once
#include "framework/math/linear_solver/sor.h"


namespace pdes
{
  /**
   * Gauss-Seidel iterative solver for Ax = b.
   * Templated on scalar type (default = types::real).
   */
  template<typename Number = types::real>
  class GaussSeidelSolver final : public SORSolver<Number>
  {
  public:
    using Result = typename LinearSolver<Number>::Result;

    GaussSeidelSolver() = default;

    explicit GaussSeidelSolver(SolverControl* control)
      : SORSolver<Number>(control, Number(1))
    {}

    std::string name() const override { return "GaussSeidelSolver"; }
  };
}
