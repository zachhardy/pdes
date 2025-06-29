#pragma once
#include "framework/types.h"
#include <vector>

namespace pdes
{
  class SolverControl
  {
  public:
    enum class Verbosity
    {
      SILENT,
      SUMMARY,
      ITERATIONS
    };

    struct Result
    {
      unsigned int iterations = 0;
      types::real residual_norm = 0.0;
      bool converged = false;
    };

    SolverControl(unsigned int max_iters,
                  types::real abs_tol,
                  types::real rel_tol = 1e-12);

    void reset();

    /**
     * Called inside solver loop with current residual norm.
     * Returns true if solver should continue.
     */
    bool check(unsigned int iter, types::real residual, Verbosity = Verbosity::SILENT);

    bool converged() const { return converged_; }
    unsigned int iterations() const { return residuals_.size(); }

    types::real initial_residual() const { return initial_residual_; }
    types::real final_residual() const { return residuals_.empty() ? 0.0 : residuals_.back(); }
    const std::vector<types::real>& residual_history() const { return residuals_; }

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
                       const types::real residual,
                       const Verbosity verbosity)
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
