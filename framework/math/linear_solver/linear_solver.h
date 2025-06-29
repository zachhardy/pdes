#pragma once
#include "framework/math/vector.h"
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/preconditioner/precondition_identity.h"
#include "framework/logger.h"


namespace pdes
{
  /**
   * A base class for iterative linear solvers.
   */
  template<typename Derived, typename VectorType = Vector<>>
  class LinearSolver
  {
  public:
    using Result = SolverControl::Result;
    using value_type = typename VectorType::value_type;

    LinearSolver() = default;
    explicit LinearSolver(SolverControl* control);

    virtual ~LinearSolver() = default;

    virtual std::string name() const = 0;

    void set_logger(const Logger& logger) { logger_ = &logger; }

    template<typename MatrixType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x) const;

    template<typename MatrixType, typename PreconditionerType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const PreconditionerType& M) const;

  protected:
    void log_iter(unsigned int iter, value_type residual_norm) const;
    void log_summary(const Result& result) const;

    SolverControl* control_;
    const Logger* logger_ = &Logger::default_logger();
  };

  /*-------------------- member functions --------------------*/

  template<typename Derived, typename VectorType>
  LinearSolver<Derived, VectorType>::LinearSolver(SolverControl* control)
    : control_(control)
  {
    static_assert(std::is_floating_point_v<value_type>,
                  "Number must be a floating point type (e.g., float, double)");
  }

  template<typename Derived, typename VectorType>
  template<typename MatrixType>
  typename LinearSolver<Derived, VectorType>::Result
  LinearSolver<Derived, VectorType>::solve(const MatrixType& A,
                                           const VectorType& b,
                                           VectorType& x) const
  {
    PreconditionIdentity identity;
    return solve(A, b, x, identity);
  }

  template<typename Derived, typename VectorType>
  template<typename MatrixType, typename PreconditionerType>
  typename LinearSolver<Derived, VectorType>::Result
  LinearSolver<Derived, VectorType>::solve(const MatrixType& A,
                                           const VectorType& b,
                                           VectorType& x,
                                           const PreconditionerType& M) const
  {
    return static_cast<const Derived *>(this)->solve(A, b, x, M);
  }

  template<typename Derived, typename VectorType>
  void
  LinearSolver<Derived, VectorType>::log_iter(const unsigned int iter,
                                              const value_type residual_norm) const
  {
    if (logger_)
      logger_->iter()
          << "iter " << iter
          << ", residual = " << residual_norm;
  }

  template<typename Derived, typename VectorType>
  void
  LinearSolver<Derived, VectorType>::log_summary(const Result& result) const
  {
    if (not logger_)
      return;

    if (result.converged)
      logger_->summary()
          << name() << " converged in " << result.iterations
          << " iterations with residual " << result.residual_norm;
    else
      logger_->warning()
          << name() << " failed to converge in " << result.iterations
          << " iterations. Final residual = " << result.residual_norm;
  }
}
