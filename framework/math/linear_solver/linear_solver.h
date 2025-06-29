#pragma once
#include "framework/math/vector.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/preconditioner/precondition_identity.h"
#include "framework/logger.h"


namespace pdes
{
  /**
   * @brief Base class for iterative linear solvers.
   *
   * @tparam Derived The derived solver class (CRTP pattern).
   * @tparam VectorType Type of the vector (default: Vector<>).
   */
  template<typename Derived, typename VectorType = Vector<>>
  class LinearSolver
  {
  public:
    using Result = SolverControl::Result;
    using value_type = typename VectorType::value_type;

    /// Constructs an uninitialized solver.
    LinearSolver() = default;

    /// Constructs a solver with a given convergence control object.
    explicit LinearSolver(SolverControl* control);

    virtual ~LinearSolver() = default;

    /// Returns the name of the solver.
    virtual std::string name() const = 0;

    /// Sets a custom logger for output.
    void set_logger(const Logger& logger) { logger_ = &logger; }

    /**
     * Solves the linear system Ax = b using the default identity preconditioner.
     *
     * @param A The system matrix.
     * @param b The right-hand side vector.
     * @param x The solution vector to be written to.
     * @return Solver result including convergence status.
     */
    template<typename MatrixType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x) const;

    /**
     * Solves the linear system Ax = b with a custom preconditioner M.
     *
     * @param A The system matrix.
     * @param b The right-hand side vector.
     * @param x The solution vector to be written to.
     * @param M The preconditioner.
     * @return Solver result including convergence status.
     */
    template<typename MatrixType, typename PreconditionerType>
    Result solve(const MatrixType& A,
                 const VectorType& b,
                 VectorType& x,
                 const PreconditionerType& M) const;

  protected:
    /// Logs one solver iteration.
    void log_iter(unsigned int iter, value_type residual_norm) const;

    /// Logs the final convergence summary.
    void log_summary(const Result& result) const;

    SolverControl* control_ = nullptr;
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
