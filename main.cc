#include "framework/math/ndarray.h"
#include "framework/math/vector.h"
#include <iostream>

#include "framework/math/matrix.h"


int
main()
{
  std::cout << "Welcome to PDEs!" << std::endl;

  using namespace pdes;

  Vector x(3, 10.0);
  x.print();

  std::cout << std::endl;

  x.scale(0.5);
  x.print();

  std::cout << std::endl;

  Matrix A(3, 2.0);
  A.print();

  std::cout << std::endl;

  const auto b = A * x;
  b.print();
}
