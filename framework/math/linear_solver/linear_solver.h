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
  template<typename Number = types::real>
  class LinearSolver
  {
  public:
    using Result = SolverControl::Result;

    LinearSolver() = default;

    explicit LinearSolver(SolverControl* control);

    virtual ~LinearSolver() = default;

    virtual std::string name() const = 0;

    void set_logger(const Logger& logger) { logger_ = &logger; }

    Result solve(const Matrix<Number>& A,
                 const Vector<Number>& b,
                 Vector<Number>& x) const;

    Result solve(const Matrix<Number>& A,
                 const Vector<Number>& b,
                 Vector<Number>& x,
                 const Preconditioner<Number>& M) const;

  protected:
    void log_iter(unsigned int iter, Number residual_norm) const;
    void log_summary(const Result& result) const;

  private:
    virtual Result _solve(const Matrix<Number>& A,
                          const Vector<Number>& b,
                          Vector<Number>& x,
                          const Preconditioner<Number>& M) const = 0;

  protected:
    SolverControl* control_;
    const Logger* logger_ = &Logger::default_logger();
  };

  /*-------------------- inline functions --------------------*/

  template<typename Number>
  LinearSolver<Number>::LinearSolver(SolverControl* control)
    : control_(control)
  {
    static_assert(std::is_floating_point_v<Number>,
                  "Number must be a floating point type (e.g., float, double)");
  }

  template<typename Number>
  typename LinearSolver<Number>::Result
  LinearSolver<Number>::solve(const Matrix<Number>& A,
                              const Vector<Number>& b,
                              Vector<Number>& x) const
  {
    PreconditionIdentity<Number> identity;
    return solve(A, b, x, identity);
  }

  template<typename Number>
  typename LinearSolver<Number>::Result
  LinearSolver<Number>::solve(const Matrix<Number>& A,
                              const Vector<Number>& b,
                              Vector<Number>& x,
                              const Preconditioner<Number>& M) const
  {
    return _solve(A, b, x, M);
  }

  template<typename Number>
  void
  LinearSolver<Number>::log_iter(const unsigned int iter,
                                 const Number residual_norm) const
  {
    if (logger_)
      logger_->iter()
          << "iter " << iter
          << ", residual = " << residual_norm;
  }

  template<typename Number>
  void
  LinearSolver<Number>::log_summary(const Result& result) const
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
