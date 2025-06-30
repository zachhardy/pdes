#pragma once
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/math/linear_solver/preconditioner/precondition_identity.h"
#include "framework/logger.h"


namespace pdes
{
  /**
   * Base class for iterative linear solvers.
   *
   * @tparam MatrixType of the matrix (default: Matrix<>).
   */
  template<typename MatrixType = Matrix<>>
  class LinearSolver
  {
  public:
    using Result = SolverControl::Result;
    using VectorType = typename MatrixType::vector_type;
    using value_type = typename MatrixType::value_type;

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
    virtual Result solve(const MatrixType& A,
                         const VectorType& b,
                         VectorType& x,
                         const Preconditioner<VectorType>& M) const = 0;

  protected:
    /// Logs one solver iteration.
    void log_iter(unsigned int iter, value_type residual_norm) const;

    /// Logs the final convergence summary.
    void log_summary(const Result& result) const;

    SolverControl* control_ = nullptr;
    const Logger* logger_ = &Logger::default_logger();
  };

  /*-------------------- member functions --------------------*/

  template<typename MatrixType>
  LinearSolver<MatrixType>::LinearSolver(SolverControl* control)
    : control_(control)
  {
    static_assert(std::is_floating_point_v<value_type>,
                  "Number must be a floating point type (e.g., float, double)");
  }

  template<typename MatrixType>
  typename LinearSolver<MatrixType>::Result
  LinearSolver<MatrixType>::solve(const MatrixType& A,
                                  const VectorType& b,
                                  VectorType& x) const
  {
    PreconditionIdentity identity;
    return solve(A, b, x, identity);
  }

  template<typename MatrixType>
  void
  LinearSolver<MatrixType>::log_iter(const unsigned int iter,
                                     const value_type residual_norm) const
  {
    if (logger_)
      logger_->iter()
          << "iter " << iter
          << ", residual = " << residual_norm;
  }

  template<typename MatrixType>
  void
  LinearSolver<MatrixType>::log_summary(const Result& result) const
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
