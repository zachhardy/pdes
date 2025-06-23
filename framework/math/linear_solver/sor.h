#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  /**
   * Successive Over-Relaxation (SOR) iterative solver for Ax = b.
   * Templated on scalar type (default = types::real).
   */
  template<typename Number = types::real>
  class SORSolver : public LinearSolver<Number>
  {
  public:
    using Result = typename LinearSolver<Number>::Result;

    explicit SORSolver(SolverControl* control);
    explicit SORSolver(SolverControl* control, Number omega);

    std::string name() const override { return "SORSolver"; }

  private:
    Result _solve(const Matrix<Number>& A,
                  const Vector<Number>& b,
                  Vector<Number>& x,
                  const Preconditioner<Number>&) const override;

    Number omega_ = Number(1.3);
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  SORSolver<Number>::SORSolver(SolverControl* control)
    : LinearSolver<Number>(control)
  {
  }

  template<typename Number>
  SORSolver<Number>::SORSolver(SolverControl* control, const Number omega)
    : LinearSolver<Number>(control),
      omega_(omega)
  {
    if (omega_ <= 0.0 || omega_ >= 2.0)
      throw std::invalid_argument("SOR relaxation parameter omega must be in (0, 2)");
  }

  template<typename Number>
  typename SORSolver<Number>::Result
  SORSolver<Number>::_solve(const Matrix<Number>& A,
                            const Vector<Number>& b,
                            Vector<Number>& x,
                            const Preconditioner<Number>&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    for (unsigned int iter = 0;; ++iter)
    {
      for (size_t i = 0; i < n; ++i)
      {
        Number sum = 0;
        const Number* row = A.begin(i);
        for (size_t j = 0; j < i; ++j)
          sum += row[j] * x(j);
        for (size_t j = i + 1; j < n; ++j)
          sum += row[j] * x(j);
        x(i) = (1.0 - omega_) * x(i) + omega_ * inv_diag[i] * (b(i) - sum);
      }

      const auto residual_norm = A.residual_norm(x, b);
      this->log_iter(iter, residual_norm);
      if (not control.check(iter, residual_norm))
      {
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }
    }
  }
}
