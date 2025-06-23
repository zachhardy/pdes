#pragma once
#include "framework/types.h"
#include "framework/math/vector.h"
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/linear_solver.h"


namespace pdes
{
  /**
   * Conjugate Gradient (CG) solver for symmetric positive-definite matrices.
   * Templated on scalar type (default = types::real).
   */
  template<typename Number = types::real>
  class CGSolver final : public LinearSolver<Number>
  {
  public:
    using Result = typename LinearSolver<Number>::Result;

    explicit CGSolver(SolverControl* control)
      : LinearSolver<Number>(control)
    {}

    std::string name() const override { return "ConjugateGradientSolver"; }

  private:
    Result _solve(const Matrix<Number>& A,
                  const Vector<Number>& b,
                  Vector<Number>& x,
                  const Preconditioner<Number>& M) const override;
  };

  template<typename Number>
  typename CGSolver<Number>::Result
  CGSolver<Number>::_solve(const Matrix<Number>& A,
                           const Vector<Number>& b,
                           Vector<Number>& x,
                           const Preconditioner<Number>& M) const
  {
    auto& control = *this->control_;

    const size_t n = b.size();
    Vector<Number> r(n), z(n), p(n), Ap(n);

    // r0 = b - Ax0
    A.vmult(x, Ap);
    r = b;
    r.add(Number(-1), Ap);

    // z0 = Minv r0
    M.vmult(r, z);

    p = z;
    auto rz_old = r.dot(z);
    for (unsigned int iter = 0;; ++iter)
    {
      A.vmult(p, Ap);
      const auto alpha = rz_old / p.dot(Ap);

      x.add(alpha, p); // x_{k+1} = x_k + \alpha_k p_k
      r.add(-alpha, Ap); // r_{k+1} = r_k - \alpha_k Ap_k
      M.vmult(r, z); // z_{k+1} = Minv r_{k+1}

      const auto rz_new = r.dot(z);
      const auto residual_norm = std::sqrt(r.dot(r));
      this->log_iter(iter, residual_norm);
      if (not control.check(iter, residual_norm))
      {
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }

      const auto beta = rz_new / rz_old;
      p.sadd(beta, Number(1), z); // p_{k+1} = z_{k+1} + \beta_k * p_k
      rz_old = rz_new;
    }
  }
}
