#pragma once
#include "framework/types.h"
#include <array>
#include <numeric>
#include <string>
#include <sstream>
#include <iomanip>

namespace pdes
{
  /// Implementation of a mesh vector in arbitrary dimensions.
  template<typename Number = types::real>
  class MeshVector final
  {
  public:
    MeshVector();
    MeshVector(const MeshVector& other);
    MeshVector(MeshVector&& other) noexcept;
    ~MeshVector() = default;

    MeshVector& operator=(const MeshVector& other);
    MeshVector& operator=(MeshVector&& other) noexcept;

    /// Constructs a MeshVector by copying another one.
    template<typename OtherNumber>
    explicit MeshVector(const MeshVector<OtherNumber>& other);

    /// Constructs a MeshVector by stealing the data from another one.
    template<typename OtherNumber>
    explicit MeshVector(MeshVector<OtherNumber>&& other);

    /// Constructs a MeshVector with a value for each dimension.
    template<typename... Args>
    explicit MeshVector(Args... coords);

    /// Copies from another MeshVector.
    template<typename OtherNumber>
    MeshVector& operator=(const MeshVector<OtherNumber>& other);
    /// Moves another MeshVector into this one.
    template<typename OtherNumber>
    MeshVector& operator=(MeshVector<OtherNumber>&& other);

    /// Returns a reference to the value of the <tt>d</tt>'th dimension.
    Number& operator()(unsigned int d);
    /// Returns the value of the <tt>d</tt>'th dimension.
    Number operator()(unsigned int d) const;

    /// Returns the length of the MeshVector.
    Number length() const;
    /// Returns the squared length of the MeshVector.
    Number length_sqr() const;

    /// Returns the distance to another MeshVector.
    Number distance(const MeshVector& other) const;
    /// Returns the squared distance to another MeshVector.
    Number distance_sqr(const MeshVector& other) const;

    /// Returns the dot product between this MeshVector and another.
    Number dot(const MeshVector& other) const;

    /// Scales the MeshVector by a scalar.
    void scale(Number value);
    /// Multiplies the MeshVector by a scalar.
    MeshVector& operator*=(Number value);
    /// Divides the MeshVector by a non-zero scalar.
    MeshVector& operator/=(Number value);

    /// Adds another MeshVector.
    MeshVector& operator+=(const MeshVector& other);
    /// Subtracts another MeshVector.
    MeshVector& operator-=(const MeshVector& other);

    /// Prints the MeshVector to an output stream.
    std::string to_string(unsigned int precision = 3,
                          bool scientific = false,
                          bool newline = true) const;

  private:
    std::array<Number, 3> coords_;

  public:
    static MeshVector unit_vector(unsigned int d);
  };

  /* -------------------- inline functions --------------------*/

  template<typename Number>
  MeshVector<Number>::MeshVector()
    : coords_{}
  {
    static_assert(std::is_floating_point_v<Number>,
                  "Point must have a floating point type.");
  }

  template<typename Number>
  MeshVector<Number>::MeshVector(const MeshVector& other)
    : MeshVector()
  {
    coords_ = other.coords_;
  }

  template<typename Number>
  MeshVector<Number>::MeshVector(MeshVector&& other) noexcept
    : MeshVector()
  {
    coords_ = std::move(other.coords_);
  }

  template<typename Number>
  MeshVector<Number>&
  MeshVector<Number>::operator=(const MeshVector& other)
  {
    coords_ = other.coords_;
    return *this;
  }

  template<typename Number>
  MeshVector<Number>&
  MeshVector<Number>::operator=(MeshVector&& other) noexcept
  {
    coords_ = std::move(other.coords_);
    return *this;
  }


  template<typename Number>
  template<typename OtherNumber>
  MeshVector<Number>::MeshVector(const MeshVector<OtherNumber>& other)
    : MeshVector()
  {
    coords_ = other.coords_;
  }

  template<typename Number>
  template<typename OtherNumber>
  MeshVector<Number>::MeshVector(MeshVector<OtherNumber>&& other)
    : MeshVector()
  {
    coords_ = std::move(other.coords_);
  }

  template<typename Number>
  template<typename... Args>
  MeshVector<Number>::MeshVector(Args... coords)
    : MeshVector()
  {
    static_assert(sizeof...(coords) > 0,
                  "Number of arguments must be greater than 0.");
    static_assert(sizeof...(coords) <= 3,
                  "Number of arguments must be 3 or fewer.");
    coords_ = {static_cast<Number>(coords)...};
  }

  template<typename Number>
  template<typename OtherNumber>
  MeshVector<Number>&
  MeshVector<Number>::operator=(const MeshVector<OtherNumber>& other)
  {
    coords_ = other.coords_;
    return *this;
  }

  template<typename Number>
  template<typename OtherNumber>
  MeshVector<Number>&
  MeshVector<Number>::operator=(MeshVector<OtherNumber>&& other)
  {
    coords_ = std::move(other.coords_);
    return *this;
  }


  template<typename Number>
  Number&
  MeshVector<Number>::operator()(const unsigned int d)
  {
    if (d >= 3)
      throw std::out_of_range("Dimension out of range.");

    return coords_[d];
  }

  template<typename Number>
  Number
  MeshVector<Number>::operator()(const unsigned int d) const
  {
    if (d >= 3)
      throw std::out_of_range("Dimension out of range.");

    return coords_[d];
  }

  template<typename Number>
  Number
  MeshVector<Number>::length() const
  {
    return std::sqrt(this->length_sqr());
  }

  template<typename Number>
  Number
  length(const MeshVector<Number>& vec)
  {
    return vec.length();
  }

  template<typename Number>
  Number
  MeshVector<Number>::length_sqr() const
  {
    return this->dot(*this);
  }

  template<typename Number>
  Number
  length_sqr(const MeshVector<Number>& vec)
  {
    return vec.length_sqr();
  }

  template<typename Number>
  Number
  MeshVector<Number>::distance(const MeshVector& other) const
  {
    return std::sqrt(this->distance_sqr(other));
  }

  template<typename Number>
  Number
  distance(const MeshVector<Number>& vec1,
           const MeshVector<Number>& vec2)
  {
    return vec1.distance(vec2);
  }

  template<typename Number>
  Number
  MeshVector<Number>::distance_sqr(const MeshVector& other) const
  {
    auto func = [](Number x, Number y) { return (x - y) * (x - y); };
    return std::inner_product(coords_.cbegin(),
                              coords_.cend(),
                              other.coords_.cbegin(),
                              0.0,
                              std::plus<Number>(),
                              func);
  }

  template<typename Number>
  Number
  distance_sqr(const MeshVector<Number>& vec1,
               const MeshVector<Number>& vec2)
  {
    return vec1.distance_sqr(vec2);
  }

  template<typename Number>
  Number
  MeshVector<Number>::dot(const MeshVector& other) const
  {
    return std::inner_product(coords_.cbegin(),
                              coords_.cend(),
                              other.coords_.cbegin(),
                              0.0);
  }

  template<typename Number>
  Number
  dot(const MeshVector<Number>& vec1,
      const MeshVector<Number>& vec2)
  {
    return vec1.dot(vec2);
  }

  template<typename Number>
  Number
  operator*(const MeshVector<Number>& vec1,
            const MeshVector<Number>& vec2)
  {
    return vec1.dot(vec2);
  }

  template<typename Number>
  void
  MeshVector<Number>::scale(const Number value)
  {
    auto func = [value](Number x) { return value * x; };
    std::transform(coords_.cbegin(), coords_.cend(), coords_.begin(), func);
  }

  template<typename Number>
  MeshVector<Number>&
  MeshVector<Number>::operator*=(const Number value)
  {
    this->scale(value);
    return *this;
  }

  template<typename Number>
  MeshVector<Number>
  operator*(const MeshVector<Number>& vec, const Number value)
  {
    MeshVector result(vec);
    result *= value;
    return result;
  }

  template<typename Number>
  MeshVector<Number>
  operator*(const Number value, const MeshVector<Number>& vec)
  {
    return vec * value;
  }

  template<typename Number>
  MeshVector<Number>
  operator-(const MeshVector<Number>& vec)
  {
    MeshVector result(vec);
    result *= Number(-1);
    return result;
  }

  template<typename Number>
  MeshVector<Number>&
  MeshVector<Number>::operator/=(Number value)
  {
    if (value == 0)
      throw std::invalid_argument("Division by zero error.");

    this->scale(Number(1) / value);
    return *this;
  }

  template<typename Number>
  MeshVector<Number>
  operator/(const MeshVector<Number>& vec, const Number value)
  {
    MeshVector result(vec);
    result /= value;
    return result;
  }

  template<typename Number>
  MeshVector<Number>&
  MeshVector<Number>::operator+=(const MeshVector& other)
  {
    auto func = [](Number x, Number y) { return x + y; };
    std::transform(coords_.cbegin(),
                   coords_.cend(),
                   other.coords_.cbegin(),
                   coords_.begin(),
                   func);
    return *this;
  }

  template<typename Number>
  MeshVector<Number>
  operator+(const MeshVector<Number>& vec1,
            const MeshVector<Number>& vec2)
  {
    MeshVector result(vec1);
    result += vec2;
    return result;
  }

  template<typename Number>
  MeshVector<Number>&
  MeshVector<Number>::operator-=(const MeshVector& other)
  {
    auto func = [](Number x, Number y) { return x - y; };
    std::transform(coords_.cbegin(),
                   coords_.cend(),
                   other.coords_.cbegin(),
                   coords_.begin(),
                   func);
    return *this;
  }

  template<typename Number>
  MeshVector<Number>
  operator-(const MeshVector<Number>& vec1,
            const MeshVector<Number>& vec2)
  {
    MeshVector result(vec1);
    result -= vec2;
    return result;
  }

  template<typename Number>
  std::string
  MeshVector<Number>::to_string(const unsigned int precision,
                                const bool scientific,
                                const bool newline) const
  {
    std::ostringstream oss;
    oss << std::setprecision(precision);
    if (scientific)
      oss << std::scientific;
    else
      oss << std::fixed;

    oss << '(' << coords_[0];
    oss << ", " << coords_[1];
    if (coords_[2] != 0.0)
      oss << ", " << coords_[2];
    oss << ')';

    if (newline)
      oss << '\n';

    return oss.str();
  }

  template<typename Number>
  MeshVector<Number>
  MeshVector<Number>::unit_vector(unsigned int d)
  {
    if (d >= 3)
      throw std::invalid_argument("Dimension out of range.");

    MeshVector result;
    result[d] = Number(1);
    return result;
  }

  template<typename Number>
  std::ostream&
  operator<<(std::ostream& os, const MeshVector<Number>& v)
  {
    os << '(' << v(0) << ", " << v(1);
    if (v(2) != 0.0)
      os << ", " << v(2);
    os << ')';
    return os;
  }
}
