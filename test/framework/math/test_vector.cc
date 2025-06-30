#include "gtest/gtest.h"

#include "framework/math/vector.h"

namespace pdes::test
{
  TEST(Vector, DefaultConstructorCreatesEmptyVector)
  {
    const Vector v;
    EXPECT_EQ(v.size(), 0);
  }

  TEST(Vector, ConstructWithSizeInitializesToZero)
  {
    Vector v(5);
    for (int i = 0; i < 5; ++i)
      EXPECT_DOUBLE_EQ(v(i), 0.0);
  }

  TEST(Vector, CanSetAndGetElements)
  {
    Vector v(3);
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    EXPECT_DOUBLE_EQ(v(0), 1.0);
    EXPECT_DOUBLE_EQ(v(1), 2.0);
    EXPECT_DOUBLE_EQ(v(2), 3.0);
  }

  TEST(Vector, AdditionAndSubtraction)
  {
    Vector<> a(3), b(3);
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    b(0) = 3.0;
    b(1) = 2.0;
    b(2) = 1.0;

    const auto c = a + b;
    EXPECT_DOUBLE_EQ(c(0), 4.0);
    EXPECT_DOUBLE_EQ(c(1), 4.0);
    EXPECT_DOUBLE_EQ(c(2), 4.0);

    const auto d = a - b;
    EXPECT_DOUBLE_EQ(d(0), -2.0);
    EXPECT_DOUBLE_EQ(d(1), 0.0);
    EXPECT_DOUBLE_EQ(d(2), 2.0);
  }

  TEST(Vector, DotProduct)
  {
    Vector a(3), b(3);
    a(0) = 1.0;
    a(1) = 2.0;
    a(2) = 3.0;
    b(0) = 4.0;
    b(1) = 5.0;
    b(2) = 6.0;
    EXPECT_DOUBLE_EQ(dot(a, b), 32.0);
  }

  TEST(Vector, OutOfBoundsThrow)
  {
    const Vector v(3);
    EXPECT_ANY_THROW(v(3)); // access beyond last index
  }

  TEST(Vector, AddMismatchedSizesThrows)
  {
    const Vector a(3);
    const Vector b(4);
    EXPECT_ANY_THROW(a + b);
  }

  TEST(Vector, DotMismatchedSizesThrows)
  {
    const Vector a(3);
    const Vector b(2);
    EXPECT_ANY_THROW(a.dot(b));
  }
}
