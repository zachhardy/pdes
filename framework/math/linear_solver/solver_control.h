#pragma once
#include "framework/types.h"
#include <vector>

namespace pdes
{
  /**
   * Convergence controller for iterative linear solvers.
   *
   * This class monitors the residual norm at each solver iteration and determines
   * whether the iteration should continue or stop based on absolute and relative
   * tolerance thresholds.
   *
   * It tracks the full residual history and exposes final convergence results.
   */
  class SolverControl
  {
  public:
    /// Summary of solver results at termination.
    struct Result
    {
      unsigned int iterations = 0; ///< Number of completed iterations
      types::real residual_norm = 0.0; ///< Final residual norm
      bool converged = false; ///< Whether convergence criteria were met
    };

    /// Constructs a convergence controller with max iterations and tolerance.
    SolverControl(unsigned int max_iters,
                  types::real abs_tol,
                  types::real rel_tol = 1e-12);

    /// Resets the internal residual history and convergence state.
    void reset();

    /**
     * Checks convergence criteria based on residual norm.
     *
     * Called once per iteration inside a solver.
     *
     * @param iter The current iteration count.
     * @param residual The current residual norm.
     * @return True if the solver should continue iterating.
     */
    bool check(unsigned int iter, types::real residual);

    /// Returns whether convergence has been achieved.
    bool converged() const { return converged_; }

    /// Returns the number of iterations completed.
    unsigned int iterations() const { return residuals_.size(); }

    /// Returns the initial residual norm (after first call to check()).
    types::real initial_residual() const { return initial_residual_; }

    /// Returns the final residual norm.
    types::real final_residual() const { return residuals_.empty() ? 0.0 : residuals_.back(); }

    /// Returns the full residual history.
    const std::vector<types::real>& residual_history() const { return residuals_; }

    /// Returns a result object summarizing the final solver state.
    Result final_state() const;

  private:
    unsigned int max_iters_;
    types::real abs_tol_;
    types::real rel_tol_;

    std::vector<types::real> residuals_;
    types::real initial_residual_ = 0.0;
    bool converged_ = false;
  };

  /*-------------------- inline functions --------------------*/

  inline
  SolverControl::SolverControl(const unsigned int max_iters,
                               const types::real abs_tol,
                               const types::real rel_tol)
    : max_iters_(max_iters),
      abs_tol_(abs_tol),
      rel_tol_(rel_tol)
  {}

  inline void
  SolverControl::reset()
  {
    residuals_.clear();
    converged_ = false;
  }

  inline bool
  SolverControl::check(const unsigned int iter,
                       const types::real residual)
  {
    if (iter >= max_iters_)
      return false;

    if (iter == 0)
      initial_residual_ = residual;

    residuals_.push_back(residual);

    if (residual < abs_tol_)
    {
      converged_ = true;
      return false;
    }

    if (initial_residual_ > 0.0 and residual / initial_residual_ < rel_tol_)
    {
      converged_ = true;
      return false;
    }
    return true;
  }

  inline SolverControl::Result
  SolverControl::final_state() const
  {
    return {iterations(), final_residual(), converged()};
  }
}
