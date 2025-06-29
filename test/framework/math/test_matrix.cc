#include <gtest/gtest.h>

#include "framework/math/matrix.h"

namespace pdes::test
{
  TEST(Matrix, ConstructWithDimensions)
  {
    const Matrix<> A(3, 4);
    std::cout << A.m() << "  " << A.n() << std::endl;
    EXPECT_EQ(A.m(), 3);
    EXPECT_EQ(A.n(), 4);
  }

  TEST(Matrix, CanSetAndGetElements)
  {
    Matrix<> A(2, 2);
    A(0, 0) = 1.0;
    A(1, 1) = 2.0;
    EXPECT_EQ(A(1, 1), 2.0);
  }

  TEST(Matrix, MultipliesVectorCorrectly)
  {
    Matrix<> A(2, 2);
    Vector<> x(2);
    A(0, 0) = 2;
    A(0, 1) = 1;
    A(1, 0) = 0;
    A(1, 1) = 3;
    x(0) = 1;
    x(1) = 2;

    auto y = A * x;
    EXPECT_EQ(y(0), 2*1 + 1*2); // = 4
    EXPECT_EQ(y(1), 0*1 + 3*2); // = 6
  }

  TEST(Matrix, MatrixMultiplyProducesExpectedResult)
  {
    Matrix<> A(2, 3);
    Matrix<> B(3, 2);

    // A = [1 2 3]
    //     [4 5 6]
    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    // B = [ 7  8]
    //     [ 9 10]
    //     [11 12]
    B(0, 0) = 7;
    B(0, 1) = 8;
    B(1, 0) = 9;
    B(1, 1) = 10;
    B(2, 0) = 11;
    B(2, 1) = 12;

    Matrix<> C = A.mmult(B);

    ASSERT_EQ(C.m(), 2);
    ASSERT_EQ(C.n(), 2);

    // C = A * B = [ 58  64 ]
    //             [139 154]
    EXPECT_DOUBLE_EQ(C(0,0), 58.0);
    EXPECT_DOUBLE_EQ(C(0,1), 64.0);
    EXPECT_DOUBLE_EQ(C(1,0), 139.0);
    EXPECT_DOUBLE_EQ(C(1,1), 154.0);
  }
}
