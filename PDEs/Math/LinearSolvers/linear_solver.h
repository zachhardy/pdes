#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include "../matrix.h"
#include "../vector.h"

namespace math
{

//######################################################################
/**
 * \brief An abstract base class for solving the linear system
 *        \f$ \boldsymbol{A} \vec{x} = \vec{b} \f$.
 */
class LinearSolver
{
protected:
  Matrix& A;  ///< The matrix \f$ \boldsymbol{A} \f$.

public:
  /// Default constructor with a matrix and right-hand side.
  LinearSolver(Matrix& matrix) : A(matrix)
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

public:
  virtual void setup() = 0; ///< Abstract setup method.
  virtual Vector solve(const Vector& b) = 0; ///< Abstract solve method.
};

}

#endif //LINEAR_SOLVER_H
