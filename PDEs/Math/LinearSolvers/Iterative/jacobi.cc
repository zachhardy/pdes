#include "jacobi.h"

#include "vector.h"
#include "sparse_matrix.h"

#include "macros.h"

#include <cmath>


using namespace pdes::Math;


LinearSolver::Jacobi::
Jacobi(const SparseMatrix& A,
       const double tolerance,
       const size_t max_iterations,
       const bool verbose) :
    IterativeSolverBase(A, tolerance, max_iterations, verbose)
{
  Assert(A.n_rows() == A.n_cols(), "Square matrix required.");
}


void
LinearSolver::Jacobi::
solve(Vector& x, const Vector& b) const
{
  size_t n = A.n_rows();
  Assert(b.size() == n, "Dimension mismatch error.");
  Assert(x.size() == n, "Dimension mismatrch error.");

  double diff;
  size_t nit;
  bool converged = false;
  Vector x_ell = x;

  //======================================== Iteration loop
  for (nit = 0; nit < max_iterations; ++nit)
  {
    diff = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
      //==================== Compute element-wise update
      double value = b[i];
      for (const auto el : A.const_row_iterator(i))
        if (el.column != i)
          value -= el.value * x_ell[el.column];
      value /= *A.diagonal(i);

      //==================== Increment difference
      diff += std::fabs(value - x_ell[i]) / std::fabs(b[i]);
      x[i] = value;
    }

    //==================== Check convergence
    x_ell = x;
    if (diff < tolerance)
    { converged = true; break; }
  }

  if (verbose)
  {
    std::stringstream ss;
    ss << "Jacobi Solver Status:\n"
       << (converged ? "  CONVERGED\n" : "  NOT CONVERGED\n")
       << (converged ? "  # Iterations: " : "  Difference: ")
       << (converged ? nit : diff) << std::endl << std::endl;
    std::cout << ss.str();
  }
}


