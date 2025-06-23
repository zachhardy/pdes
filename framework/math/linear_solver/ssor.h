#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"
#include <vector>
#include <stdexcept>
#include <string>

namespace pdes
{
  /**
   * Symmetric Successive Over-Relaxation (SSOR) solver for Ax = b.
   * Performs a forward and backward sweep per iteration.
   */
  template<typename Number = types::real>
  class SSORSolver final : public LinearSolver<Number>
  {
  public:
    using Result = typename LinearSolver<Number>::Result;

    explicit SSORSolver(SolverControl* control);
    explicit SSORSolver(SolverControl* control, Number omega);

    std::string name() const override { return "SSORSolver"; }

  private:
    Result _solve(const Matrix<Number>& A,
                  const Vector<Number>& b,
                  Vector<Number>& x,
                  const Preconditioner<Number>&) const override;

    Number omega_ = Number(1.3);
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  SSORSolver<Number>::SSORSolver(SolverControl* control)
    : LinearSolver<Number>(control)
  {
  }

  template<typename Number>
  SSORSolver<Number>::SSORSolver(SolverControl* control, const Number omega)
    : LinearSolver<Number>(control),
      omega_(omega)
  {
    if (omega_ <= 0.0 || omega_ >= 2.0)
      throw std::invalid_argument("SSOR relaxation parameter omega must be in (0, 2)");
  }


  template<typename Number>
  typename SSORSolver<Number>::Result
  SSORSolver<Number>::_solve(const Matrix<Number>& A,
                             const Vector<Number>& b,
                             Vector<Number>& x,
                             const Preconditioner<Number>&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    for (unsigned int iter = 0;; ++iter)
    {
      // Forward sweep
      for (size_t i = 0; i < n; ++i)
      {
        Number sum = 0;
        const auto row = A.begin(i);
        for (size_t j = 0; j < i; ++j)
          sum += row[j] * x(j);
        for (size_t j = i + 1; j < n; ++j)
          sum += row[j] * x(j);
        x(i) = (1 - omega_) * x(i) + omega_ * inv_diag[i] * (b(i) - sum);
      }

      // Backward sweep
      for (size_t i = n; i-- > 0;)
      {
        Number sum = 0;
        const auto row = A.begin(i);
        for (size_t j = 0; j < i; ++j)
          sum += row[j] * x(j);
        for (size_t j = i + 1; j < n; ++j)
          sum += row[j] * x(j);
        x(i) = (1 - omega_) * x(i) + omega_ * inv_diag[i] * (b(i) - sum);
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
