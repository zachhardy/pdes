#pragma once
#include "framework/math/linear_solver/linear_solver.h"
#include "framework/math/linear_solver/preconditioner/preconditioner.h"

namespace pdes
{
  template<typename MatrixType = Matrix<>>
  class BiCGStabSolver : public LinearSolver<MatrixType>
  {
  public:
    using value_type = typename LinearSolver<MatrixType>::value_type;
    using VectorType = typename LinearSolver<MatrixType>::VectorType;
    using Result = typename LinearSolver<MatrixType>::Result;

    explicit BiCGStabSolver(SolverControl* control) : LinearSolver<MatrixType>(control) {}

    using LinearSolver<MatrixType>::solve;

    /**
     * Solves Ax = b using the CG method with preconditioner M.
     *
     * @param A Symmetric positive-definite system matrix.
     * @param b Right-hand side vector.
     * @param x Solution vector (initial guess and result).
     * @param M Preconditioner to apply on residuals.
     * @return Solver result with convergence status.
     */
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const Preconditioner<MatrixType>& M) const override;

    std::string name() const override { return "BiCGStab"; }
  };

  template<typename MatrixType>
  typename BiCGStabSolver<MatrixType>::Result
  BiCGStabSolver<MatrixType>::solve(const MatrixType& A,
                                    const VectorType& b,
                                    VectorType& x,
                                    const Preconditioner<MatrixType>& M) const
  {
    auto& control = *this->control_;
    const auto n = b.size();

    // Create vectors
    VectorType r(n), r0(n);
    VectorType p(n), v(n), s(n), t(n);
    VectorType phat(n), shat(n);

    // Compute initial residual: r = b - A*x
    A.residual(x, b, r);
    r0 = r; // choose r0 as the initial shadow residual

    // Initialize scalars
    value_type alpha(1.0), omega(1.0);
    value_type rho(1.0), rho_old(1.0);

    // Initialize some vectors
    p = r;
    v = 0.0;

    for (unsigned int iter = 0;; ++iter)
    {
      // Compute v = A * phat = A * M * p
      M.vmult(p, phat);
      A.vmult(phat, v);

      // Compute alpha =  rho / (r0, v)
      const auto r0v = r0.dot(v);
      if (r0v == value_type(0))
        throw std::runtime_error("BiCGStab breakdown: r0v == 0");

      alpha = rho_old / r0v;

      // Compute intermediate solution:
      // x = x + alpha * phat
      x.add(alpha, phat);

      // Compute intermediate residual:
      // s = r - alpha * v
      s = r;
      s.add(-alpha, v);

      // Check for early convergence
      if (not control.check(iter, s.l2_norm()))
      {
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }

      // Compute t = A * shat = A * M * s
      M.vmult(s, shat);
      A.vmult(shat, t);

      // Compute omega = (t, s) / (t, t)
      const value_type tt = t.dot(t);
      if (tt == value_type(0))
        throw std::runtime_error("BiCGStab breakdown: tt == 0");

      omega = t.dot(s) / tt;

      // Update solution: x = x + omega * shat
      x.add(omega, shat);

      // Update residual: r = s - omega * t
      r = s;
      r.add(-omega, t);

      // Log and check convergence
      const auto residual_norm = r.l2_norm();
      this->log_iter(iter, residual_norm);
      if (not control.check(iter, residual_norm))
      {
        const auto result = control.final_state();
        this->log_summary(result);
        return result;
      }

      // Compute rho = (r0, r), check for breakdown
      rho = r0.dot(r);
      if (rho == value_type(0))
        throw std::runtime_error("BiCGStab breakdown: rho = 0");

      // Update search direction:
      // beta = rho/rho_old * alpha/omega
      // p = r + beta * (p - omega * v)
      const auto beta = rho / rho_old * alpha / omega;
      p.sadd(beta, -beta * omega, v);
      p.add(r);

      rho_old = rho;
    }
  }
} // namespace pdes
