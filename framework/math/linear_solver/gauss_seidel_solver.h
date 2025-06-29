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
    using Base = SORSolver<VectorType>;
    using Result = typename Base::Result;
    using value_type = typename VectorType::value_type;

    GaussSeidelSolver() = default;
    explicit GaussSeidelSolver(SolverControl* control) : Base(control, value_type(1)) {}

    std::string name() const override { return "GaussSeidelSolver"; }
  };
}
