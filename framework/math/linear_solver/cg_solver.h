#pragma once
#include "framework/types.h"
#include "framework/math/vector.h"
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/linear_solver.h"


namespace pdes
{
  /**
   * @brief Conjugate Gradient (CG) solver for symmetric positive-definite systems.
   *
   * This solver implements the classical CG method, an efficient Krylov subspace
   * technique for solving large sparse symmetric positive-definite systems.
   *
   * It supports preconditioning and returns a result struct containing convergence info.
   *
   * @tparam VectorType The vector type to use (default: Vector<>).
   */
  template<typename VectorType = Vector<>>
  class CGSolver final : public LinearSolver<CGSolver<VectorType>, VectorType>
  {
  public:
    using Base = LinearSolver<CGSolver, VectorType>;
    using Result = typename Base::Result;
    using value_type = typename VectorType::value_type;

    /// Constructs a CG solver with a given convergence controller.
    explicit CGSolver(SolverControl* control) : Base(control) {}

    /// Returns the name of the solver.
    std::string name() const override { return "ConjugateGradientSolver"; }

    using Base::solve;

    /**
     * Solves Ax = b using the CG method with preconditioner M.
     *
     * @param A Symmetric positive-definite system matrix.
     * @param b Right-hand side vector.
     * @param x Solution vector (initial guess and result).
     * @param M Preconditioner to apply on residuals.
     * @return Solver result with convergence status.
     */
    template<typename MatrixType, typename PreconditionerType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const PreconditionerType& M) const;
  };

  /*-------------------- member functions --------------------*/

  template<typename VectorType>
  template<typename MatrixType, typename PreconditionerType>
  typename CGSolver<VectorType>::Result
  CGSolver<VectorType>::solve(const MatrixType& A,
                              const VectorType& b,
                              VectorType& x,
                              const PreconditionerType& M) const
  {
    auto& control = *this->control_;

    const size_t n = b.size();
    Vector<value_type> r(n), z(n), p(n), Ap(n);

    // r0 = b - Ax0
    A.vmult(x, Ap);
    r = b;
    r.add(value_type(-1), Ap);

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
      p.sadd(beta, value_type(1), z); // p_{k+1} = z_{k+1} + \beta_k * p_k
      rz_old = rz_new;
    }
  }
}
