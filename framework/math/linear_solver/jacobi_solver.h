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
  template<typename VectorType = Vector<>>
  class JacobiSolver final : public LinearSolver<JacobiSolver<VectorType>, VectorType>
  {
  public:
    using Base = LinearSolver<JacobiSolver, VectorType>;
    using Result = typename Base::Result;
    using value_type = typename VectorType::value_type;

    JacobiSolver() = default;
    explicit JacobiSolver(SolverControl* control) : Base(control) {}

    std::string name() const override { return "JacobiSolver"; }

    using Base::solve;

    template<typename MatrixType, typename PreconditionerType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const PreconditionerType&) const;
  };

  /*-------------------- member functions --------------------*/

  template<typename VectorType>
  template<typename MatrixType, typename PreconditionerType>
  typename JacobiSolver<VectorType>::Result
  JacobiSolver<VectorType>::solve(const MatrixType& A,
                                  const VectorType& b,
                                  VectorType& x,
                                  const PreconditionerType&) const
  {
    auto& control = *this->control_;
    const auto inv_diag = internal::extract_inv_diagonal(A, name());

    const auto n = b.size();
    VectorType x_old(n, value_type(0));
    for (unsigned int iter = 0;; ++iter)
    {
      x_old = x;

      // Compute the updated iterate
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (size_t i = 0; i < n; ++i)
      {
        value_type sum = 0;
        const value_type* row = A.begin(i);
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

