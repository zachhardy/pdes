#pragma once

namespace pdes
{
  /**
   * @brief Identity preconditioner.
   *
   * This is a no-op preconditioner that simply copies the input vector to the output.
   * It is used as a default when no preconditioning is desired.
   */
  class PreconditionIdentity
  {
  public:
    /// Applies z = r with no modification.
    template<typename VectorType>
    static void vmult(const VectorType& src, VectorType& dst) { dst = src; }

    /// Returns the name of the preconditioner.
    static std::string name() { return "PreconditionIdentity"; }
  };
}
