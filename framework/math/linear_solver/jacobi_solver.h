#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/util.h"


namespace pdes
{
  /**
   * A simple Jacobi iterative solver for Ax = b.
   * Templated on scalar type (default = types::real).
   */
  template<typename Number = types::real>
  class JacobiSolver final : public LinearSolver<Number>
  {
  public:
    using Result = typename LinearSolver<Number>::Result;

    JacobiSolver() = default;

    explicit JacobiSolver(SolverControl* control)
      : LinearSolver<Number>(control)
    {}

    std::string name() const override { return "JacobiSolver"; }

  private:
    Result _solve(const Matrix<Number>& A,
                  const Vector<Number>& b,
                  Vector<Number>& x,
                  const Preconditioner<Number>&) const override;
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  typename JacobiSolver<Number>::Result
  JacobiSolver<Number>::_solve(const Matrix<Number>& A,
                               const Vector<Number>& b,
                               Vector<Number>& x,
                               const Preconditioner<Number>&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    Vector<Number> x_old(n, Number(0));
    for (unsigned int iter = 0;; ++iter)
    {
      x_old = x;

      // Compute the updated iterate
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < n; ++i)
      {
        Number sum = 0;
        const Number* row = A.begin(i);
        for (size_t j = 0; j < n; ++j)
          if (j != i)
            sum += row[j] * x_old(j);
        x(i) = inv_diag[i] * (b(i) - sum);
      }

      const auto residual_norm = A.residual_norm(x, b);
      this->log_iter(iter, residual_norm);

      if (not control.check(iter, residual_norm))
      {
        // Return result if converged
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }
    }
  }
}

