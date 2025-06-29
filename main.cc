#include "framework/math/vector.h"
#include "framework/math/matrix.h"
#include "framework/math/linear_solver/cg.h"
#include "framework/math/linear_solver/preconditioner/precondition_ssor.h"
#include "framework/math/linear_solver/solver_control.h"
#include "framework/logger.h"
#include <iostream>

#include "framework/math/linear_solver/jacobi.h"


int
main()
{
  std::cout << "Welcome to PDEs!" << std::endl;

  using namespace pdes;

  constexpr size_t n = 10; // number of interior nodes
  Matrix A(n, n, 0.0);
  Vector b(n, 0.0);
  Vector x(n, 0.0); // initial guess = 0

  // Assemble 1D Laplacian matrix (Dirichlet BCs: u(0) = u(1) = 0)
  constexpr auto h = 1.0 / n;
  for (size_t i = 0; i < n; ++i)
  {
    A(i, i) = 2.0;
    if (i > 0)
      A(i, i - 1) = -1.0;
    if (i < n - 1)
      A(i, i + 1) = -1.0;
    b(i) = h * h; // constant source term
  }

  // Preconditioned solve
  const PreconditionSSOR<> M(&A);
  SolverControl control(100, 1.0e-8);
  CGSolver<> solver(&control);
  const Logger logger(std::cout, LogLevel::DEBUG);
  solver.set_logger(logger);
  auto [iterations, residual, converged] = solver.solve(A, b, x, M);

  std::cout << "Solution: " << x.to_string(5) << '\n';
}
