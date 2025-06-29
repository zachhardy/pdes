#pragma once
#include "framework/math/linear_solver/solver_control.h"
#include <iostream>
#include <iomanip>

namespace pdes::test::utils
{
  template<typename SolverType, typename MatrixType,
           typename VectorType, typename PreconditionerType>
  unsigned int test_convergence(const std::string& label,
                                const MatrixType& A,
                                const VectorType& b,
                                const PreconditionerType& M,
                                const double tol = 1.0e-10,
                                const unsigned int max_iters = 1000)
  {
    SolverControl control(max_iters, tol);
    SolverType solver(&control);

    VectorType x(b.size(), 0.0);
    solver.solve(A, b, x, M);

    std::cout << std::setw(18) << std::left << label
              << " | iter: " << std::setw(4) << control.iterations()
              << " | final ||r||: " << control.final_residual() << "\n";

    return control.iterations();
  }
}