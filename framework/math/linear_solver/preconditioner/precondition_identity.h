#pragma once

namespace pdes
{
  class PreconditionIdentity
  {
  public:
    template<typename VectorType>
    static void vmult(const VectorType& src, VectorType& dst);

    static std::string name() { return "PreconditionIdentity"; }
  };

  /*-------------------- member functions --------------------*/

  template<typename VectorType>
  void
  PreconditionIdentity::vmult(const VectorType& src, VectorType& dst)
  {
    dst = src;
  }
}
