#include "gtest/gtest.h"

#include "framework/math/vector.h"

namespace pdes::test
{
  TEST(Vector, DefaultConstructorCreatesEmptyVector)
  {
    const Vector<> v;
    EXPECT_EQ(v.size(), 0);
  }

  TEST(Vector, ConstructWithSizeInitializesToZero)
  {
    Vector<> v(5);
    for (int i = 0; i < 5; ++i)
      EXPECT_EQ(v(i), 0.0);
  }

  TEST(Vector, CanSetAndGetElements)
  {
    Vector<> v(3);
    v(0) = 1.0;
    v(1) = 2.0;
    v(2) = 3.0;
    EXPECT_EQ(v(1), 2.0);
  }

  TEST(Vector, AdditionAndSubtraction)
  {
    Vector<> a(3), b(3);
    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    b(0) = 3;
    b(1) = 2;
    b(2) = 1;

    auto c = a + b;
    EXPECT_EQ(c(0), 4.0);
    EXPECT_EQ(c(2), 4.0);

    auto d = a - b;
    EXPECT_EQ(d(0), -2.0);
    EXPECT_EQ(d(1), 0.0);
  }

  TEST(Vector, DotProduct)
  {
    Vector<> a(3), b(3);
    a(0) = 1;
    a(1) = 2;
    a(2) = 3;
    b(0) = 4;
    b(1) = 5;
    b(2) = 6;

    EXPECT_EQ(dot(a, b), 1*4 + 2*5 + 3*6);
  }
}
