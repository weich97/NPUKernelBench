#include <tuple>
#include <kernel_operator.h>
#include <type_traits>
#include "lib/matmul_intf.h"

#define __TLA_REQUIRES(...)   typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

namespace tla {
template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <class T>
using remove_cvref_t = typename remove_cvref<T>::type;

template <class T, class = void>
struct tuple_size;

template <class T>
struct tuple_size<T, std::void_t<typename std::tuple_size<T>::type>>
    : std::integral_constant<size_t, std::tuple_size<T>::value> {};

template <class T>
constexpr size_t tuple_size_v = tuple_size<T>::value;

} // end namespace tla

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

template <uint32_t ALIGN, typename T>
__forceinline__ [aicore]
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

template <class T>
__forceinline__ [aicore]
constexpr T RoundUp(const T &val, const T align)
{
    return (val + align - 1) / align * align;
}

template <uint32_t ALIGN, typename T>
__forceinline__ [aicore]
constexpr T RoundDown(const T val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return val / ALIGN * ALIGN;
}

template <class T>
__forceinline__ [aicore]
constexpr T RoundDown(const T val, const T align)
{
    return val / align * align;
}

template <uint32_t DIVISOP, typename T>
__forceinline__ [aicore]
constexpr T CeilDiv(const T dividend)
{
    static_assert(DIVISOP != 0, "DIVISOP must not be 0");
    return (dividend + DIVISOP - 1) / DIVISOP;
}

template <class T>
__forceinline__ [aicore]
constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;

constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;

}  // namespace Catlass

namespace tla {
template <class T, class U,
          __TLA_REQUIRES(std::is_arithmetic<T>::value &&
                         std::is_arithmetic<U>::value)>
__forceinline__ [aicore] constexpr
auto
min(T const& t, U const& u) {
    return t < u ? t : u;
}
} // namespace tla

namespace Catlass {

template <
    int RANK_,
    class Index_ = uint32_t,
    class LongIndex_ = int64_t
>
struct Coord {
public:
    static const int RANK = RANK_;
    using Index = Index_;
    using LongIndex = LongIndex_;

    __forceinline__ [aicore] constexpr
    explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    __forceinline__ [aicore] constexpr
    Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    // Constructs frrom an array of integers
    __forceinline__ [aicore]
    int Argmin() const
    {
        int i = 0;
        for (int j = 1; j < RANK; ++j) {
            if (idx[j] < idx[i]) {
                i = j;
            }
        }
        return i;
    }

    // Returns the index of the dimension with greatest value
    __forceinline__ [aicore]
    int Argmax() const
    {
        int i = 0;
        for (int j = 1; j < RANK; ++j) {
            if (idx[j] > idx[i]) {
                i = j;
            }
        }
        return i;
    }

    // Returns true if Coord is non-zero
    __forceinline__ [aicore]
    explicit operator bool() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i]) {
                return true;
            }
        }
        return false;
    }

    // Return true if Coord is uniformly zero.
    __forceinline__ [aicore]
    bool operator!() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i]) {
                return false;
            }
        }
        return true;
    }

    // Element-wise addition
    __forceinline__ [aicore]
    Coord operator+(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] + b.idx[i];
        }
        return c;
    }

    // Add a scalar to each element
    __forceinline__ [aicore]
    Coord operator+(const Index val) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] + val;
        }
        return c;
    }

    // Element-wise subtraction
    __forceinline__ [aicore]
    Coord operator-(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] - b.idx[i];
        }
        return c;
    }

    // Subtract a scalar from each element
    __forceinline__ [aicore]
    Coord operator-(Index const val) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c.idx[i] = idx[i] - val;
        }
        return c;
    }

    // Element-wise multiply
    __forceinline__ [aicore]
    Coord operator*(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] * b.idx[i];
        }
        return c;
    }

    // Element-wise division
    __forceinline__ [aicore]
    Coord operator/(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] / b.idx[i];
        }
        return c;
    }

    // Element-wise mod
    __forceinline__ [aicore]
    Coord operator%(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; i++) {
            c.idx[i] = idx[i] % b.idx[i];
        }
        return c;
    }

    // In-place addition
    __forceinline__ [aicore]
    Coord &operator+=(Coord const &b)
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] += b.idx[i];
        }
        return *this;
    }

    // In-place equal
    __forceinline__ [aicore]
    bool operator==(Coord const &b) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != b.idx[i]) {
                return false;
            }
        }
        return true;
    }

    // In-place equal
    __forceinline__ [aicore]
    bool operator==(Index const val) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != val) {
                return false;
            }
        }
        return true;
    }

    // Member acces operator
    __forceinline__ [aicore]
    Index &operator[](int dim)
    {
        return idx[dim];
    }

    __forceinline__ [aicore]
    Index const &operator[](int dim) const
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    __forceinline__ [aicore]
    Index &At()
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    __forceinline__ [aicore]
    Index &At(int dim)
    {
        return idx[dim];
    }

    // Gets the index of a given Coord element
    template <int DIM>
    __forceinline__ [aicore]
    Index const &At() const
    {
        return idx[DIM];
    }

    // Access via index; may limit unrolling potential
    __forceinline__ [aicore]
    Index const &At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    __forceinline__ [aicore]
    auto GetCoordByAxis() const
    {
        Index idx_[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{idx_};
    }

    __forceinline__ [aicore]
    static Coord Min(Coord const &a, Coord const &b)
    {
        Coord res;
        for (int i = 0; i < RANK; ++i) {
            res[i] = a[i] < b[i] ? a[i] : b[i];
        }
        return res;
    }

private:
    Index idx[RANK];
};

template <class T>
__forceinline__ [aicore] constexpr
Coord<1, T> MakeCoord(T dim0)
{
    T values[1] = {dim0};
    return Coord<1, T>(values);
}

template <class T>
__forceinline__ [aicore] constexpr
Coord<2, T> MakeCoord(T dim0, T dim1)
{
    T values[2] = {dim0, dim1};
    return Coord<2, T>(values);
}

template <class T>
__forceinline__ [aicore] constexpr
Coord<3, T> MakeCoord(T dim0, T dim1, T dim2)
{
    T values[3] = {dim0, dim1, dim2};
    return Coord<3, T>(values);
}

/// Helper to make a 4-element coordinate
template <class T>
__forceinline__ [aicore] constexpr
Coord<4, T> MakeCoord(T dim0, T dim1, T dim2, T dim3)
{
    T values[4] = {dim0, dim1, dim2, dim3};
    return Coord<4, T>(values);
}
template <
    uint32_t M_ = 1,
    uint32_t N_ = 1,
    uint32_t K_ = 1
>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

    static constexpr int64_t MN = M * N;
    static constexpr int64_t MK = M * K;
    static constexpr int64_t KN = N * K;
    static constexpr int64_t MNK = M * N * K;

    static constexpr int64_t COUNT = MNK;

    /// Returns a Coord object
    __forceinline__ [aicore]
    static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }

    __forceinline__ [aicore]
    static Coord<2> ToCoordMN()
    {
        return MakeCoord(M, N);
    }

    __forceinline__ [aicore]
    static Coord<2> ToCoordMK()
    {
        return MakeCoord(M, K);
    }

    __forceinline__ [aicore]
    static Coord<2> ToCoordKN()
    {
        return MakeCoord(K, N);
    }
};

struct GemmCoord : public Coord<3, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<3, Index>;
    static constexpr int M_INDEX = 0;
    static constexpr int N_INDEX = 1;
    static constexpr int K_INDEX = 2;

    __forceinline__ [aicore]
    GemmCoord() {}

    __forceinline__ [aicore]
    GemmCoord(Coord<3, Index> const &coord) : Base(coord) {}

    __forceinline__ [aicore]
    GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    __forceinline__ [aicore]
    Index const &m() const { return this->At(M_INDEX); }
    __forceinline__ [aicore]
    Index &m() { return this->At(M_INDEX); }

    __forceinline__ [aicore]
    Index const &n() const { return this->At(N_INDEX); }
    __forceinline__ [aicore]
    Index &n() { return this->At(N_INDEX); }

    __forceinline__ [aicore]
    Index const &k() const { return this->At(K_INDEX); }
    __forceinline__ [aicore]
    Index &k() { return this->At(K_INDEX); }

    __forceinline__ [aicore]
    auto GetCoordMN() const
    {
        return this->GetCoordByAxis<M_INDEX, N_INDEX>();
    }

    __forceinline__ [aicore]
    auto GetCoordMK() const
    {
        return this->GetCoordByAxis<M_INDEX, K_INDEX>();
    }

    __forceinline__ [aicore]
    auto GetCoordKN() const
    {
        return this->GetCoordByAxis<K_INDEX, N_INDEX>();
    }
};

struct MatrixCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<2, Index>;
    using LongIndex = typename Base::LongIndex;
    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;

    __forceinline__ [aicore]
    MatrixCoord() {}

    __forceinline__ [aicore]
    MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}

    __forceinline__ [aicore]
    MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

    /// Helper to construct from a row and column, which are LongIndex based
    __forceinline__ [aicore]
    MatrixCoord(LongIndex row, LongIndex column) : Base(MakeCoord(Index(row), Index(column))) {}

    /// Returns the row of the coordinate
    __forceinline__ [aicore]
    Index const &row() const { return this->At(ROW_INDEX); }
    __forceinline__ [aicore]
    Index &row() { return this->At(ROW_INDEX); }

    __forceinline__ [aicore]
    Index const &column() const { return this->At(COLUMN_INDEX); }
    __forceinline__ [aicore]
    Index &column() { return this->At(COLUMN_INDEX); }

    /// Element-wise addition
    __forceinline__ [aicore]
    MatrixCoord operator+(Base const &b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    /// In-place addition
    __forceinline__ [aicore]
    MatrixCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }
};

} // namespace Catlass

namespace Catlass::layout {

struct VectorLayout {
public:
    static constexpr int RANK = 1;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;
    using TensorCoord = Coord<RANK, Index>;
public:
    __forceinline__ [aicore]
    VectorLayout(Index size = 0) : shape_(MakeCoord(size)), stride_(MakeCoord(LongIndex(1))) {}

    __forceinline__ [aicore]
    VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    __forceinline__ [aicore]
    static VectorLayout MakeLayoutInUb(TensorCoord const &tileShape)
    {
        return VectorLayout{RoundUp<BYTE_PER_BLK / sizeof(Element)>(tileShape[0])};
    }

    __forceinline__ [aicore]
    LongIndex GetOffset(TensorCoord const &coord) const
    {
        return stride_[0] * coord[0];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    VectorLayout GetTileLayout(TensorCoord const &tileShape) const
    {
        return VectorLayout(tileShape, stride());
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    Shape shape_;
    Stride stride_;
};

struct RowMajor {
public:
    static constexpr int RANK = 2;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;
public:
    __forceinline__ [aicore]
    RowMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(cols), LongIndex(1))) {}

    /// Constructor
    __forceinline__ [aicore]
    RowMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(ldm, LongIndex(1))) {}

    /// Ctor
    __forceinline__ [aicore]
    RowMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    template <class Element>
    __forceinline__ [aicore]
    static RowMajor MakeLayoutInUb(MatrixCoord const &shape)
    {
        return RowMajor(shape.row(), shape.column(), RoundUp<BYTE_PER_C0 / sizeof(Element)>(shape.column()));
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_[0] + LongIndex(coord.column());
    }

    __forceinline__ [aicore]
    RowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return RowMajor(tileShape, stride());
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }
private:
    Shape shape_;
    Stride stride_;
};

struct zN {
public:
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;
public:
    __forceinline__ [aicore] constexpr
    zN(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    __forceinline__ [aicore] constexpr
    zN(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    __forceinline__ [aicore] constexpr
    static zN MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zN(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL,
                  1,
                  rowsRound * ELE_NUM_PER_C0);
    }

    __forceinline__ [aicore]
    static zN MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return zN(shape.row(),
                  shape.column(),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.row()),
                  C0_NUM_PER_FRACTAL,
                  CeilDiv<C0_NUM_PER_FRACTAL>(shape.column()),
                  C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL,
                  1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.row()) * C0_NUM_PER_FRACTAL);
    }

    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    zN GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2))
        );
        return zN(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    OrgShape orgShape_;
    Shape shape_;
    Stride stride_;
};

struct zZ {
public:
    static constexpr int RANK = 4;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;
    using Shape = Coord<RANK, Index>;
    using Stride = Coord<RANK, LongIndex>;
public:
    __forceinline__ [aicore] constexpr
    zZ(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    __forceinline__ [aicore] constexpr
    zZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    __forceinline__ [aicore] constexpr
    static zZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return zZ(orgRows,
                  orgCols,
                  C0_NUM_PER_FRACTAL,
                  rowsRound / C0_NUM_PER_FRACTAL,
                  ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  colsRound * C0_NUM_PER_FRACTAL,
                  1,
                  ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    OrgShape orgShape_;
    Shape shape_;
    Stride stride_;
};

struct nZ {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    __forceinline__ [aicore] constexpr
    nZ(Index orgRows = 0,                 /// Number of rows of origin matrices
       Index orgCols = 0,                 /// Number of cols of origin matrices
       Index rowsInFractal = 0,           /// Number of rows inside the fractal
       Index rowsByFractal = 0,           /// number of rows by the fractal
       Index colsInFractal = 0,           /// number of cols inside the fractal
       Index colsByFractal = 0,           /// number of cols by the fractal
       LongIndex strideRowsInFractal = 0, /// number of elements between adjacent rows inside the fractal
       LongIndex strideRowsByFractal = 0, /// number of elements between adjacent fractal rows
       LongIndex strideColsInFractal = 0, /// number of elements between adjacent cols inside the fractal
       LongIndex strideColsByFractal = 0) /// number of elements between adjacent fractal cols
        : orgShape_(MakeCoord(orgRows, orgCols)),
          shape_(MakeCoord(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal)),
          stride_(MakeCoord(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)) {}

    /// Ctor
    __forceinline__ [aicore] constexpr
    nZ(OrgShape orgShape, Shape shape, Stride stride) : orgShape_(orgShape), shape_(shape), stride_(stride) {}

    /// Make the layout of a coordinate (row, column)
    template <class Element>
    __forceinline__ [aicore] constexpr
    static nZ MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return nZ(orgRows,
                  orgCols,
                  ELE_NUM_PER_C0,
                  rowsRound / ELE_NUM_PER_C0,
                  C0_NUM_PER_FRACTAL,
                  colsRound / C0_NUM_PER_FRACTAL,
                  1,
                  colsRound * ELE_NUM_PER_C0,
                  ELE_NUM_PER_C0,
                  ELE_NUM_PER_FRACTAL);
    }

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_[0] * stride_[1] + LongIndex(coord.column()) / shape_[2] * stride_[3] +
            (LongIndex(coord.row()) % shape_[0]) * stride_[0] + (LongIndex(coord.column()) % shape_[2]) * stride_[2];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    nZ GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        auto tileShape = MakeCoord(
            shape(0), CeilDiv(tileOriShape.row(), shape(0)),
            shape(2), CeilDiv(tileOriShape.column(), shape(2))
        );
        return nZ(tileOriShape, tileShape, stride());
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index orgShape(int idx) const
    {
        return orgShape_[idx];
    }

    /// Returns the origin shape of the layout
    __forceinline__ [aicore]
    typename OrgShape::Index &orgShape(int idx)
    {
        return orgShape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape shape() const
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    Shape &shape()
    {
        return shape_;
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index shape(int idx) const
    {
        return shape_[idx];
    }

    /// Returns the shape of the layout
    __forceinline__ [aicore]
    typename Shape::Index &shape(int idx)
    {
        return shape_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride stride() const
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    Stride &stride()
    {
        return stride_;
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index stride(int idx) const
    {
        return stride_[idx];
    }

    /// Returns the stride of the layout
    __forceinline__ [aicore]
    typename Stride::Index &stride(int idx)
    {
        return stride_[idx];
    }

private:
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

} // namespace Catlass::layout

namespace Catlass::Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct LocalTensorBufferBase {
public:
    template <class Element = half>
    __forceinline__ [aicore]
    AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const
    {
        return tensor[offset].template ReinterpretCast<Element>();
    }
protected:
    __forceinline__ [aicore]
    LocalTensorBufferBase() = default;
    AscendC::LocalTensor<uint8_t> tensor;
};

template <class ArchTag, AscendC::TPosition Position>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
        GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
        tensor = tbufA1.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
public:
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
        GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
        tensor = tbufA2.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
public:
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
        GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
        tensor = tbufB2.Get<uint8_t>();
    }
};

template <>
struct LocalTensorBuffer<Arch::AtlasA2, AscendC::TPosition::C2> : LocalTensorBufferBase {
public:
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
        GetTPipePtr()->InitBuffer(tbufC2, Arch::AtlasA2::BIAS_SIZE);
        tensor = tbufC2.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
        GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
        tensor = tbufCO1.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
public:
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};

template<class ArchTag>
struct Resource {
public:
    AscendC::TPipe pipe;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;

    __forceinline__ [aicore]
    Resource()
    {
        pipe.Destroy();
    }
};

} // namespace Catlass::Arch

namespace Catlass::Gemm {
template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = POSITION_;
};

template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadAtlasA2Base<false>;

template <bool ENABLE_UNIT_FLAG_ = false>
struct MmadAtlasA2PingpongBias : public MmadAtlasA2  {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
};

}  // namespace Catlass::Gemm

namespace Catlass::Gemm::helper {

template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>, "Unsupported element accumulator selector, can not find the specialization.");
};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template<class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>, "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>, "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class GmBiasType, class ElementAccumulator>
struct L1BiasTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBiasType>, "Unsupported layout selector, can not find the specialization.");
};

template<class ElementAccumulator>
struct L1BiasTypeSelector<void, ElementAccumulator> {
    using GMBiasType = void;
    using L1BiasType = void;
    using L0BiasType = void;
};

template<class Element, class ElementAccumulator>
struct L1BiasTypeSelector<Gemm::GemmType<Element, layout::VectorLayout>, ElementAccumulator> {
    using GMBiasType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>;
    using L1BiasType = Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>;
    using L0BiasType = Gemm::GemmType<ElementAccumulator, layout::VectorLayout, AscendC::TPosition::C2>;
};

template<class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};

template<class Element>
struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

} // namespace Catlass::Gemm::helper

namespace Catlass::Gemm::Tile {

template <class ArchTag, class GmType, class L1Type = void>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};

template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    __forceinline__ [aicore]
    CopyGmToL1() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::Nd2NzParams intriParams;
        intriParams.ndNum = 1;
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(0) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(0);
            intriParams.srcDValue = layoutSrc.stride(0);
            intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(0); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(0)], intriParams);
            }
        }
    }
};

template <class ArchTag, class Element>
struct CopyGmToL1<ArchTag, Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::GM>,
    Gemm::GemmType<Element, layout::VectorLayout, AscendC::TPosition::A1>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    __forceinline__ [aicore]
    CopyGmToL1() {};
    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C0;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    __forceinline__ [aicore]
    CopyL1ToL0A() {};
    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
    __forceinline__ [aicore]
    CopyL1ToL0B() {};
    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.startIndex = 0;
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<ELE_NUM_PER_C0>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / ELE_NUM_PER_FRACTAL;
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / ELE_NUM_PER_FRACTAL - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToBT {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to biasTable buffer, can not find the specialization.");
};

template<class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToBT<ArchTag, Catlass::Gemm::GemmType<ElementSrc, layout::VectorLayout, AscendC::TPosition::A1>,
    Catlass::Gemm::GemmType<ElementDst, layout::VectorLayout, AscendC::TPosition::C2>>{
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_C2 =  BYTE_PER_C2 / sizeof(ElementSrc);

    __forceinline__ [aicore]
    CopyL1ToBT(){}

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<ElementDst> dstTensor,
        AscendC::LocalTensor<ElementSrc> srcTensor,
        LayoutDst layoutDst, LayoutSrc layoutSrc)
    {
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C2;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

enum class ScaleGranularity { NO_QUANT = 0 };

template <class ArchTag, class ElementSrc, class ElementDst, ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <>
struct CopyL0CToGmQuantMode<Catlass::Arch::AtlasA2, float, half, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <class ArchTag, class ElementAccumulator, class GmType,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <class ElementAccumulator_, class ElementDst_, bool ReluEnable_>
struct CopyL0CToGm<Catlass::Arch::AtlasA2, ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>, ScaleGranularity::NO_QUANT, ReluEnable_> {
    using ArchTag = Catlass::Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = Catlass::layout::zN;
    using LayoutDst = Catlass::layout::RowMajor;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst,
        ScaleGranularity::NO_QUANT>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    __forceinline__ [aicore]
    void operator()(AscendC::GlobalTensor<ElementDst> const &dst, AscendC::LocalTensor<ElementSrc> const &src,
        LayoutDst const &dstLayout, LayoutSrc const &srcLayout, uint8_t unitFlag = 0)
    {
        AscendC::FixpipeParamsV220 intriParams;
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <class ArchTag, class AType_, class BType_, class BiasType_>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    __forceinline__ [aicore]
    TileMmad() {}

    __forceinline__ [aicore]
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
         AscendC::LocalTensor<ElementA> const &l0ATensor,
         AscendC::LocalTensor<ElementB> const &l0BTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = initC;
        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, mmadParams);
        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }

    __forceinline__ [aicore]
    void operator()(AscendC::LocalTensor<ElementAccumulator> const &l0CTensor,
         AscendC::LocalTensor<ElementA> const &l0ATensor,
         AscendC::LocalTensor<ElementB> const &l0BTensor,
         AscendC::LocalTensor<ElementAccumulator> const &l0BiasTensor,
         uint32_t m, uint32_t n, uint32_t k,
         bool initC = true, uint8_t unitFlag = 0)
    {
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.unitFlag = unitFlag;
        mmadParams.cmatrixInitVal = false;
        AscendC::Mmad(l0CTensor, l0ATensor, l0BTensor, l0BiasTensor, mmadParams);
        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};

template <class ArchTag, class AType, class BType, class CType, class BiasType = void>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyGmToL1<ArchTag, typename BiasTypeSelector::GMBiasType, typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>, void,
        Gemm::Tile::CopyL1ToBT<ArchTag, typename BiasTypeSelector::L1BiasType, typename BiasTypeSelector::L0BiasType>>;
};

} // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::Block {

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          class BiasType = void,
          class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
          class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    bool ENABLE_UNIT_FLAG_,
    class L1TileShape_,
    class L0TileShape_,
    class AType_,
    class BType_,
    class CType_,
    class BiasType_,
    class TileCopy_,
    class TileMmad_
>
struct BlockMmad <
    MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG_>,
    L1TileShape_,
    L0TileShape_,
    AType_,
    BType_,
    CType_,
    BiasType_,
    TileCopy_,
    TileMmad_
> {
public:
    using DispatchPolicy = MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementBias = typename BiasType_::Element;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyGmToL1Bias = typename TileCopy_::CopyGmToL1Bias;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator = typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using CopyL1ToBT = typename TileCopy_::CopyL1ToBT;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;
    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");
    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    __forceinline__ [aicore]
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        uint32_t l1BiasOffset = l1BufAddrStart + L1A_SIZE * STAGES + L1B_SIZE * STAGES;
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_SIZE * i);
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + STAGES;
            l0AEventList[i] = i;
            l0BEventList[i] = i + STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        l0CTensor = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(0);
        l1BiasTensor = resource.l1Buf.template GetBufferByByte<ElementBias>(l1BiasOffset);
        l0BiasTensor = resource.btBuf.template GetBufferByByte<ElementAccumulator>(0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    __forceinline__ [aicore]
    ~BlockMmad()
    {
        for (uint32_t i = 0; i < STAGES; i++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    __forceinline__ [aicore]
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementBias> const &gmBias,
        GemmCoord const &actualShape)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutBiasInL1 = layout::VectorLayout(L1TileShape::N);
        auto layoutBiasInL0 = layout::VectorLayout(L0TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));
        uint32_t kActual = min(actualShape.k(), L1TileShape::K);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
        auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
        copyGmToL1A(l1ATensorList[l1ListId], gmA, layoutAInL1, layoutTileA);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
        auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
        copyGmToL1B(l1BTensorList[l1ListId], gmB, layoutBInL1, layoutTileB);
        auto layoutTileBias = layout::VectorLayout(actualShape.n());
        copyGmToL1Bias(l1BiasTensor, gmBias, layoutBiasInL1, layoutTileBias);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);
        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};
            if (kLoopIdx < kTileCount - 1) {
                uint32_t kLoopIdxNext = kLoopIdx + 1;
                kActualNext = (kLoopIdxNext < kTileCount - 1) ? L1TileShape::K : (actualShape.k() - kLoopIdxNext * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, kLoopIdxNext * L1TileShape::K};
                MatrixCoord gmTileBOffset{kLoopIdxNext * L1TileShape::K, 0};
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensorList[l1ListIdNext], gmA[layoutA.GetOffset(gmTileAOffset)], layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensorList[l1ListIdNext], gmB[layoutB.GetOffset(gmTileBOffset)], layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);
            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ? L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);
                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ? L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) { AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]); }
                    copyL1ToL0A(l0ATensorList[l0AListId], l1ATensorList[l1ListId][layoutAInL1.GetOffset(l1AOffset)], layoutAInL0, layoutAInL1);
                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) { AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]); }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ? L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        if ((kPartIdx == 0) && (nPartIdx == 0)) { AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]); }
                        copyL1ToL0B(l0BTensorList[l0BListId], l1BTensorList[l1ListId][layoutBInL1.GetOffset(l1BOffset)], layoutBInL0, layoutBInL1);
                        copyL1ToBT(l0BiasTensor, l1BiasTensor, layoutBiasInL0, layoutBiasInL1);
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) { AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]); }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else { unitFlag = 0b10; }
                        }
                        if (initC) {
                            tileMmad(l0CTensor[layoutInL0C.GetOffset(l0COffset)], l0ATensorList[l0AListId], l0BTensorList[l0BListId], l0BiasTensor, mPartActual, nPartActual, kPartActual, initC, unitFlag);
                        } else {
                            tileMmad(l0CTensor[layoutInL0C.GetOffset(l0COffset)], l0ATensorList[l0AListId], l0BTensorList[l0BListId], mPartActual, nPartActual, kPartActual, initC, unitFlag);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                        l0BListId = (l0BListId + 1 < STAGES) ? (l0BListId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                    l0AListId = (l0AListId + 1 < STAGES) ? (l0AListId + 1) : 0;
                }
            }
            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());
        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }
protected:
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementBias> l1BiasTensor;
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;
    AscendC::LocalTensor<ElementAccumulator> l0BiasTensor;
    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];
    uint32_t l1ListId{0};
    uint32_t l0AListId{0};
    uint32_t l0BListId{0};
    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyGmToL1Bias copyGmToL1Bias;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL1ToBT copyL1ToBT;
    CopyL0CToGm copyL0CToGm;
};

struct GemmIdentityBlockSwizzle {
    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    __forceinline__ [aicore]
    GemmIdentityBlockSwizzle() {}

    __forceinline__ [aicore]
    GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }

    __forceinline__ [aicore]
    uint32_t GetCoreLoops() const
    {
        return loopsMN.row() * loopsMN.column();
    }

    __forceinline__ [aicore]
    GemmCoord GetBlockCoord(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        // SwizzleDirection is 0 (Zn) and SwizzleOffset is 3
        uint32_t SwizzleOffset = 3;
        uint32_t tileBlockLoop = CeilDiv(loopsMN.row(), SwizzleOffset);
        uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.column());
        uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.column());
        uint32_t nRow = SwizzleOffset;
        if (tileBlockIdx == tileBlockLoop - 1) {
            nRow = loopsMN.row() - SwizzleOffset * tileBlockIdx;
        }
        uint32_t mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
        uint32_t nIdx = inTileBlockIdx / nRow;
        if (tileBlockIdx % 2 == 1) {
            nIdx = loopsMN.column() - nIdx - 1;
        }
        return GemmCoord{mIdx, nIdx, 0};
    }

    __forceinline__ [aicore]
    GemmCoord GetActualBlockShape(GemmCoord blockCoord)
    {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        uint32_t kActual = problemShape.k();
        return GemmCoord{mActual, nActual, kActual};
    }
};

}  // namespace Catlass::Gemm::Block

namespace Catlass::Gemm::Kernel {

template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_>
class MatmulBias {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias;
    using BlockScheduler = BlockScheduler_;

    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrBias;

        __forceinline__ [aicore]
        Params(
            GemmCoord const &problemShape_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrBias_
        ) : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_), ptrBias(ptrBias_) {}
    };

    __forceinline__ [aicore]
    MatmulBias() {}

    template <int32_t CORE_TYPE = g_coreType>
    __forceinline__ [aicore]
    void operator()(Params const &params);

    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);
        AscendC::GlobalTensor<ElementBias> gmBias;
        gmBias.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);
            int64_t gmOffsetBias = blockCoord.n() * L1TileShape::N;

            blockMmad(
                gmA[gmOffsetA], params.layoutA,
                gmB[gmOffsetB], params.layoutB,
                gmC[gmOffsetC], params.layoutC,
                gmBias[gmOffsetBias],
                actualBlockShape);
        }
    }
};

} // namespace Catlass::Gemm::Kernel

using namespace Catlass;
using namespace matmul;

template <
    class LayoutA,
    class LayoutB,
    class LayoutBias,
    class LayoutC
>
__forceinline__ [aicore]
void MatmulBias(
    GemmCoord problemShape,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmBias, LayoutBias layoutBias,
    GM_ADDR gmC, LayoutC layoutC
)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = Gemm::MmadAtlasA2PingpongBias<true>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;
    using BiasType = Gemm::GemmType<half, LayoutBias>;
    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType>;
    using BlockEpilogue = void;
    using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle;
    using MatmulKernel = Gemm::Kernel::MatmulBias<BlockMmad, BlockEpilogue, BlockScheduler>;
    typename MatmulKernel::Params params{problemShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, gmBias};
    MatmulKernel matmul;
    matmul(params);
}

extern "C" __global__ __aicore__ void matmul_bias(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t m = tiling_data.m;
    uint32_t k = tiling_data.k;
    uint32_t n = tiling_data.n;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    using LayoutBias = layout::VectorLayout;

    GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};
    LayoutBias layoutBias{n};

    MatmulBias<LayoutA, LayoutB, LayoutBias, LayoutC>(
        problemShape, a, layoutA, b, layoutB, bias, layoutBias, c, layoutC
    );
}