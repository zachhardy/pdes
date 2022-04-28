#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "Math/matrix.h"
#include "Math/vector.h"

namespace math
{

enum class LinearSolverType
{
  LU = 0,
  CHOLESKY = 1
};

//######################################################################
/**
 * \brief An abstract base class for solving the linear system
 *        \f$ \boldsymbol{A} \vec{x} = \vec{b} \f$.
 */
class LinearSolver
{
protected:
  bool initialized = false;
  Matrix<double>& A;

public:
  LinearSolver(Matrix<double>& matrix) : A(matrix)
  {
    if (A.n_rows() != A.n_cols())
    {
      std::stringstream err;
      err << "LinearSystem::" << __FUNCTION__ << ": "
          << "Invalid inputs. The matrix must be square and of the same "
          << "dimension as the right-hand side vector.";
      throw std::runtime_error(err.str());
    }
  }

  void set_matrix(Matrix<double>& matrix) { A = matrix; initialized = false; }

public:
  /** Abstract setup method. */
  virtual void setup() = 0;

  /** Abstract solve method. */
  virtual Vector<double> solve(const Vector<double>& b) = 0;
};

}
#endif //LINEAR_SOLVER_H
