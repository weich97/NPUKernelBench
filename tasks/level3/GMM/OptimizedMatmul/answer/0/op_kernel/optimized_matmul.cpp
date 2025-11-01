#include <kernel_operator.h>
#include "lib/matmul_intf.h"
#include <type_traits>

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

// =================================================================================================
// General Utility Functions
// =================================================================================================

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

// =================================================================================================
// Catlass - Core Data Structures & Constants
// =================================================================================================

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
template <
    int RANK_,                         ///< Logical rank of coordinate
    class Index_ = uint32_t,        ///< Index type used for each dimension
    class LongIndex_ = int64_t      ///< Long index type used for linear offsets
>
struct Coord {
public:
    // Number of elements in Coord
    static const int RANK = RANK_;

    // Index typen used to store elements
    using Index = Index_;

    // Type used to represent linear offsets
    using LongIndex = LongIndex_;

    // Default ctor initializes uniformly
    __forceinline__ [aicore] constexpr
    explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    // Constructs from an array of integers
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

    // Member access operator
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

/// Helper to make a 2-element coordinate
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
template <uint32_t M_ = 1, uint32_t N_ = 1, uint32_t K_ = 1>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;
};

struct GemmCoord : public Coord<3, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<3, Index>;
    static constexpr int M_INDEX = 0;
    static constexpr int N_INDEX = 1;
    static constexpr int K_INDEX = 2;

    __forceinline__ [aicore] GemmCoord() {}
    __forceinline__ [aicore] GemmCoord(Coord<3, Index> const &coord) : Base(coord) {}
    __forceinline__ [aicore] GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    __forceinline__ [aicore] Index const &m() const { return this->At(M_INDEX); }
    __forceinline__ [aicore] Index &m() { return this->At(M_INDEX); }
    __forceinline__ [aicore] Index const &n() const { return this->At(N_INDEX); }
    __forceinline__ [aicore] Index &n() { return this->At(N_INDEX); }
    __forceinline__ [aicore] Index const &k() const { return this->At(K_INDEX); }
    __forceinline__ [aicore] Index &k() { return this->At(K_INDEX); }

    __forceinline__ [aicore] auto GetCoordMN() const {
        Index idx_[2]{this->At(M_INDEX), this->At(N_INDEX)};
        return Coord<2, Index, int64_t>{idx_};
    }
};

struct MatrixCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<2, Index>;
    using LongIndex = typename Base::LongIndex;
    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;

    __forceinline__ [aicore] MatrixCoord() {}
    __forceinline__ [aicore] MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}
    __forceinline__ [aicore] MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

    __forceinline__ [aicore] Index const &row() const { return this->At(ROW_INDEX); }
    __forceinline__ [aicore] Index const &column() const { return this->At(COLUMN_INDEX); }
};

} // namespace Catlass

// =================================================================================================
// Catlass Layouts
// =================================================================================================

namespace Catlass::layout {
struct VectorLayout {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 1;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Shape vector
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

    /// Logical coordinate
    using TensorCoord = Coord<RANK, Index>;

public:
    // Methods

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
    /// Stride data member
    Shape shape_;
    Stride stride_;
};

/// Mapping function for row-major matrices
struct RowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
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

    /// Returns the layout of a tile.
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
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for col-major matrices
struct ColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    // Methods

    /// Constructor
    __forceinline__ [aicore]
    ColumnMajor(Index rows = 0, Index cols = 0)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), LongIndex(rows))) {}

    /// Constructor
    __forceinline__ [aicore]
    ColumnMajor(Index rows, Index cols, LongIndex ldm)
        : shape_(MakeCoord(rows, cols)), stride_(MakeCoord(LongIndex(1), ldm)) {}

    /// Ctor
    __forceinline__ [aicore]
    ColumnMajor(Shape shape, Stride stride) : shape_(shape), stride_(stride) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_[1];
    }

    /// Returns the layout of a tile.
    __forceinline__ [aicore]
    ColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ColumnMajor(tileShape, stride());
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
    //
    // Data members
    //

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for zN matrices which is row-major inside fractal and col-major between fractal
struct zN {
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
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for zN matrices which is row-major inside fractal and row-major between fractal
struct zZ {
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
    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for nZ matrices which is col-major inside fractal and row-major between fractal
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

/// Mapping function for padding rowmajor matrices
/// A special data layout designed to improve the efficiency of matrix operations in non-512B aligned scenarios.
/// This layout is row-major within blocks and also row-major between blocks.
struct PaddingRowMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    __forceinline__ [aicore]
    PaddingRowMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0) :
        orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(blockRows, CeilDiv(orgRows, blockRows), blockCols, CeilDiv(orgCols, blockCols))),
        stride_(MakeCoord((LongIndex)blockCols, (LongIndex)blockRows * (LongIndex)RoundUp(orgCols, blockCols),
        (LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols)) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1]
            + (LongIndex)coord.column() / blockCols * stride_[3]
            + (LongIndex)coord.row() % blockRows * stride_[0]
            + (LongIndex)coord.column() % blockCols;
    }

    __forceinline__ [aicore]
    PaddingRowMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return PaddingRowMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
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
    //
    // Data members
    //

    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};

/// Mapping function for padding columnmajor matrices
/// A special data layout designed to improve the efficiency of matrix operations in non-512B aligned scenarios.
/// This layout is column-major within blocks and also column-major between blocks.
struct PaddingColumnMajor {
public:
    /// Logical rank of tensor
    static constexpr int RANK = 4;

    /// Logical rank of orgshape
    static constexpr int ORG_SHAPE_RANK = 2;

    /// Index type used for coordinates
    using Index = uint32_t;

    /// Long index type used for offsets
    using LongIndex = int64_t;

    /// Logical coordinate
    using OrgShape = Coord<ORG_SHAPE_RANK, Index>;

    /// Logical coordinate
    using Shape = Coord<RANK, Index>;

    /// Stride vector
    using Stride = Coord<RANK, LongIndex>;

public:
    /// Constructor
    __forceinline__ [aicore]
    PaddingColumnMajor(Index orgRows = 0, Index orgCols = 0, Index blockRows = 0, Index blockCols = 0) :
        orgShape_(MakeCoord(orgRows, orgCols)),
        shape_(MakeCoord(blockRows, CeilDiv(orgRows, blockRows), blockCols, CeilDiv(orgCols, blockCols))),
        stride_(MakeCoord((LongIndex)1, (LongIndex)blockRows * (LongIndex)blockCols, (LongIndex)blockRows,
        (LongIndex)RoundUp(orgRows, blockRows) * (LongIndex)blockCols)) {}

    /// Returns the offset of a coordinate in linear memory.
    /// Assumes coordinate has convention (row, column)
    __forceinline__ [aicore]
    LongIndex GetOffset(MatrixCoord const &coord) const
    {
        LongIndex blockRows = (LongIndex)shape_[0];
        LongIndex blockCols = (LongIndex)shape_[2];
        return (LongIndex)coord.row() / blockRows * stride_[1]
            + (LongIndex)coord.column() / blockCols * stride_[3]
            + (LongIndex)coord.row() % blockRows
            + (LongIndex)coord.column() % blockCols * stride_[2];
    }

    __forceinline__ [aicore]
    PaddingColumnMajor GetTileLayout(MatrixCoord const &tileShape) const
    {
        return PaddingColumnMajor(tileShape.row(), tileShape.column(), shape_[0], shape_[2]);
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
    //
    // Data members
    //

    /// Origin Shape data member
    OrgShape orgShape_;

    /// Shape data member
    Shape shape_;

    /// Stride data member
    Stride stride_;
};
} // namespace Catlass::layout

// =================================================================================================
// Catlass Architecture & Memory
// =================================================================================================
namespace Catlass::Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

using FlagID = uint16_t;
struct CrossCoreFlag {
    __forceinline__ [aicore] CrossCoreFlag(FlagID id) : id(id) {}
    FlagID id;
};

template <uint8_t MODE, pipe_t PIPE>
__forceinline__ [aicore] void CrossCoreBarrier() {
    constexpr FlagID flagId = (g_coreType == AscendC::AIC) ? 9 : 8;
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flagId);
    AscendC::CrossCoreWaitFlag(flagId);
}

template <uint8_t MODE, pipe_t PIPE>
__forceinline__ [aicore] void CrossCoreSetFlag(CrossCoreFlag &flag){
    AscendC::CrossCoreSetFlag<MODE, PIPE>(flag.id);
}

__forceinline__ [aicore] void CrossCoreWaitFlag(CrossCoreFlag &flag){
    AscendC::CrossCoreWaitFlag(flag.id);
}

struct LocalTensorBufferBase {
protected:
    __forceinline__ [aicore] LocalTensorBufferBase() = default;
    AscendC::LocalTensor<uint8_t> tensor;
public:
    template <class Element = half>
    __forceinline__ [aicore] AscendC::LocalTensor<Element> GetBufferByByte(const uint32_t offset) const {
        return tensor[offset].template ReinterpretCast<Element>();
    }
};

template <class ArchTag, AscendC::TPosition Position> struct LocalTensorBuffer;

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
    __forceinline__ [aicore] LocalTensorBuffer() {
        AscendC::TBuf<AscendC::TPosition::A1> tbufA1;
        GetTPipePtr()->InitBuffer(tbufA1, ArchTag::L1_SIZE);
        tensor = tbufA1.Get<uint8_t>();
    }
};
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> : LocalTensorBufferBase {
    __forceinline__ [aicore] LocalTensorBuffer() {
        AscendC::TBuf<AscendC::TPosition::A2> tbufA2;
        GetTPipePtr()->InitBuffer(tbufA2, ArchTag::L0A_SIZE);
        tensor = tbufA2.Get<uint8_t>();
    }
};
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> : LocalTensorBufferBase {
    __forceinline__ [aicore] LocalTensorBuffer() {
        AscendC::TBuf<AscendC::TPosition::B2> tbufB2;
        GetTPipePtr()->InitBuffer(tbufB2, ArchTag::L0B_SIZE);
        tensor = tbufB2.Get<uint8_t>();
    }
};
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
    __forceinline__ [aicore] LocalTensorBuffer() {
        AscendC::TBuf<AscendC::TPosition::CO1> tbufCO1;
        GetTPipePtr()->InitBuffer(tbufCO1, ArchTag::L0C_SIZE);
        tensor = tbufCO1.Get<uint8_t>();
    }
};
template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> : LocalTensorBufferBase {
    __forceinline__ [aicore] LocalTensorBuffer() {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};

template<class ArchTag>
struct Resource {
    AscendC::TPipe pipe;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;
    __forceinline__ [aicore] Resource() { pipe.Destroy(); }
};

} // namespace Catlass::Arch

// =================================================================================================
// Catlass GEMM - Generic Structures
// =================================================================================================

namespace Catlass::Gemm {

template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
};

namespace Block {
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    __forceinline__ [aicore] GemmIdentityBlockSwizzle(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_) {
        loopsMN = CeilDiv(MatrixCoord(problemShape.GetCoordMN()), tileMN);
    }
    __forceinline__ [aicore] uint32_t GetCoreLoops() const {
        return loopsMN.row() * loopsMN.column();
    }
    __forceinline__ [aicore] GemmCoord GetBlockCoord(uint32_t taskIdx) {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        if constexpr (SwizzleDirection == 0) { // Zn
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
        } else { // Nz
            uint32_t tileBlockLoop = CeilDiv(loopsMN.column(), SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopsMN.row());
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopsMN.row());
            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopsMN.column() - SwizzleOffset * tileBlockIdx;
            }
            uint32_t mIdx = inTileBlockIdx / nCol;
            uint32_t nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopsMN.row() - mIdx - 1;
            }
            return GemmCoord{mIdx, nIdx, 0};
        }
    }
    __forceinline__ [aicore] GemmCoord GetActualBlockShape(GemmCoord blockCoord) {
        uint32_t mActual = (blockCoord.m() == (loopsMN.row() - 1)) ?
            (problemShape.m() - blockCoord.m() * tileMN.row()) : tileMN.row();
        uint32_t nActual = (blockCoord.n() == (loopsMN.column() - 1)) ?
            (problemShape.n() - blockCoord.n() * tileMN.column()) : tileMN.column();
        return GemmCoord{mActual, nActual, problemShape.k()};
    }
};
} // namespace Block

// --- Dispatch Policies ---
template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2 = MmadAtlasA2Base<false>;
using MmadAtlasA2Async = MmadAtlasA2Base<true>;

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA2Preload : public MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};
} // namespace Catlass::Gemm

// =================================================================================================
// Catlass GEMM - Epilogue & Tile Operations
// =================================================================================================

namespace Catlass::Epilogue::Tile {
template <
    class ArchTag,
    class GmType
>
struct CopyGm2Ub {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to ub, can not find the specialization.");
};
template <typename Element>
struct CopyGm2Ub<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutSrc = layout::RowMajor;
    using LayoutDst = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_BLK = BYTE_PER_BLK / sizeof(Element);

    __forceinline__ [aicore]
    CopyGm2Ub() = default;

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutSrc.shape(0),
            layoutSrc.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) * sizeof(Element),
            (layoutDst.stride(0) - layoutDst.shape(1)) / ELE_NUM_PER_BLK,
            0
        );
        AscendC::DataCopyPadExtParams<Element> padParams(false, 0, 0, 0);
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
    };
};

template <
    class ArchTag,
    class GmType
>
struct CopyUb2Gm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy ub to gm, can not find the specialization.");
};
template <typename Element>
struct CopyUb2Gm<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::RowMajor;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    __forceinline__ [aicore]
    CopyUb2Gm() = default;

    __forceinline__ [aicore]
    void operator()(
        AscendC::GlobalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        layout::RowMajor const &layoutDst,
        layout::RowMajor const &layoutSrc)
    {
        AscendC::DataCopyExtParams dataCopyParams(
            layoutDst.shape(0),
            layoutDst.shape(1) * sizeof(Element),
            (layoutSrc.stride(0) - layoutSrc.shape(1)) / ELE_NUM_PER_C0,
            (layoutDst.stride(0) - layoutDst.shape(1)) * sizeof(Element),
            0
        );
        AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams);
    }
};
} // namespace Catlass::Epilogue::Tile

namespace Catlass::Gemm::helper {
template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>,
        "Unsupported element accumulator selector, can not find the specialization.");
};
template<> struct ElementAccumulatorSelector<half, half> { using ElementAccumulator = float; };

template<class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
};
template<class Element> struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};
template<class Element> struct L1ATypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};
template<class Element> struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};
template<class Element> struct L1BTypeSelector<Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class GmBiasType, class ElementAccumulator>
struct L1BiasTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBiasType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class ElementAccumulator>
struct L1BiasTypeSelector<void, ElementAccumulator> {
    using GMBiasType = void;
    using L1BiasType = void;
    using L0BiasType = void;
};

template<class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported align helper, can not find the specialization.");
};
template<class Element> struct L1AlignHelper<Element, layout::RowMajor> {
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t N_ALIGNED = BYTE_PER_C0 / sizeof(Element);
};
template<class Element> struct L1AlignHelper<Element, layout::PaddingRowMajor> {
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t N_ALIGNED = BYTE_PER_C0 / sizeof(Element);
};
} // namespace Catlass::Gemm::helper

namespace Catlass::Gemm::Tile {
template <
    class ArchTag,
    /// GemmType for matrix operand
    class GmType,
    class L1Type = void
>
struct CopyGmToL1 {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1, can not find the specialization.");
};
/// Partial specialization for AtlasA2, RowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::RowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::RowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

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

    // layoutSrc must be the layout of one of the src matrices
    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::GlobalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc,
        uint32_t ndNum, uint32_t srcNdMatrixStride,
        uint32_t dstNzNStride, uint32_t dstNzMatrixStride,
        uint32_t dstNzC0Stride)
    {
        AscendC::Nd2NzParams intriParams;

        intriParams.nValue = layoutSrc.shape(0);
        intriParams.dValue = layoutSrc.shape(1);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = dstNzNStride;
        intriParams.dstNzC0Stride = dstNzC0Stride;
        if (srcNdMatrixStride < STRIDE_LIMIT) {
            intriParams.ndNum = ndNum;
            intriParams.srcNdMatrixStride = srcNdMatrixStride;
            intriParams.dstNzMatrixStride = dstNzMatrixStride;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.ndNum = 1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
            for (uint32_t i = 0; i < ndNum; i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * srcNdMatrixStride], intriParams);
            }
        }
    }
};

/// Partial specialization for AtlasA2, PaddingRowMajor in and zN out.
template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::PaddingRowMajor>> {
    using LayoutDst = layout::zN;
    using LayoutSrc = layout::PaddingRowMajor;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);

    // Mehtods

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
        intriParams.dValue = layoutSrc.orgShape(1);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(3) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        intriParams.nValue = layoutSrc.orgShape(0);
        intriParams.srcDValue = layoutSrc.stride(0);
        intriParams.dstNzNStride = layoutDst.stride(0) / ELE_NUM_PER_C0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0A {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};
template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

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

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};
/// Partial specialization for zN in and nZ out.
template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);

    // Methods

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

enum class ScaleGranularity {
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <
    class ArchTag,
    class ElementSrc,
    class ElementDst,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT
>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

// CopyL0CToGm cast fp32 to fp16
template <>
struct CopyL0CToGmQuantMode<
    Catlass::Arch::AtlasA2,
    float, half,
    ScaleGranularity::NO_QUANT
> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};
template <
    class ArchTag,
    class ElementAccumulator,
    class GmType,
    ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT,
    bool ReluEnable = false
>
struct CopyL0CToGm {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm, can not find the specialization.");
};

template <
    class ElementAccumulator_,
    class ElementDst_,
    bool ReluEnable_
>
struct CopyL0CToGm<Catlass::Arch::AtlasA2,
                   ElementAccumulator_,
                   Gemm::GemmType<ElementDst_, layout::RowMajor>,
                   ScaleGranularity::NO_QUANT,
                   ReluEnable_>
{
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

        // Fixpipe layout information
        intriParams.nSize = dstLayout.shape(1);
        intriParams.mSize = dstLayout.shape(0);
        intriParams.srcStride = srcLayout.stride(3) / srcLayout.stride(0);
        intriParams.dstStride = dstLayout.stride(0);

        // Fixpipe auxiliary arguments
        intriParams.quantPre = quantPre;
        intriParams.reluEn = reluEn;
        intriParams.unitFlag = unitFlag;

        // Call AscendC Fixpipe
        AscendC::Fixpipe<ElementDst, ElementSrc, AscendC::CFG_ROW_MAJOR>(dst, src, intriParams);
    }
};

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
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
        LayoutDst layoutDst, LayoutSrc layoutSrc
    ){
        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = layoutDst.shape(0) / ELE_NUM_PER_C2;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        AscendC::DataCopy(dstTensor, srcTensor, intriParams);
    }
};

template <
    /// Tag indicating architecture
    class ArchTag_,
    /// GemmType for A matrix operand
    class AType_,
    /// GemmType type for B matrix operand
    class BType_,
    /// GemmType type for Bias operand
    class BiasType_
>
struct TileMmad {
    using ElementA = typename AType_::Element;
    using ElementB = typename BType_::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    // Methods

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
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<typename AType_::Layout, layout::ColumnMajor>) {
            mmadParams.kDirectionAlign = true;
        }

        AscendC::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      mmadParams);

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
        if constexpr (std::is_same_v<ElementA, float> && std::is_same_v<typename AType_::Layout, layout::ColumnMajor>) {
            mmadParams.kDirectionAlign = true;
        }

        AscendC::Mmad(l0CTensor,
                      l0ATensor,
                      l0BTensor,
                      l0BiasTensor,
                      mmadParams);

        const uint32_t PIPE_M_BARRIER_THRESHOLD = 10;
        if ((m / C0_NUM_PER_FRACTAL) * (n / C0_NUM_PER_FRACTAL) < PIPE_M_BARRIER_THRESHOLD) {
            AscendC::PipeBarrier<PIPE_M>();
        }
    }
};
template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopy {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    using CopyGmToL1A = Gemm::Tile::CopyGmToL1<ArchTag, AType>;
    using CopyGmToL1B = Gemm::Tile::CopyGmToL1<ArchTag, BType>;
    using CopyL1ToL0A = Gemm::Tile::CopyL1ToL0A<
        ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0B = Gemm::Tile::CopyL1ToL0B<
        ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGm = Gemm::Tile::CopyL0CToGm<ArchTag, ElementAccumulator, CType>;
    using BiasTypeSelector = helper::L1BiasTypeSelector<BiasType, ElementAccumulator>;
    using CopyGmToL1Bias = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyGmToL1<ArchTag,
            typename BiasTypeSelector::GMBiasType,
            typename BiasTypeSelector::L1BiasType>>;
    using CopyL1ToBT = std::conditional_t<std::is_same_v<BiasType, void>,
        void,
        Gemm::Tile::CopyL1ToBT<ArchTag,
            typename BiasTypeSelector::L1BiasType,
            typename BiasTypeSelector::L0BiasType>>;
};

} // namespace Catlass::Gemm::Tile

// =================================================================================================
// Catlass GEMM - Block Level Operations
// =================================================================================================

namespace Catlass::Gemm::Block {
template <
    class DispatchPolicy,
    class L1TileShape,
    class L0TileShape,
    class AType,
    class BType,
    class CType,
    class BiasType = void,
    class TileCopy = Gemm::Tile::TileCopy<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
    class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>
>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

template <
    bool ENABLE_UNIT_FLAG_,
    bool ENABLE_SHUFFLE_K_,
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
    MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>,
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
    // Type Aliases
    using DispatchPolicy = MmadAtlasA2Preload<ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;
    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using TileMmad = TileMmad_;
    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using L1AAlignHelper = Gemm::helper::L1AlignHelper<ElementA, LayoutA>;
    using L1BAlignHelper = Gemm::helper::L1AlignHelper<ElementB, LayoutB>;

    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;
    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L1A_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;

    // Check LayoutC
    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    // Check L1TileShape
    static_assert((L1A_SIZE * STAGES + L1B_SIZE * STAGES) <= ArchTag::L1_SIZE, "L1TileShape exceeding the L1 space!");

    // Check L0TileShape
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static_assert((L0A_TILE_SIZE * STAGES) <= L0A_SIZE, "L0TileShape exceeding the L0A space!");
    static_assert((L0B_TILE_SIZE * STAGES) <= L0B_SIZE, "L0TileShape exceeding the L0B space!");

    static_assert(L1TileShape::M == L0TileShape::M && L1TileShape::N == L0TileShape::N,
        "The situation where the basic blocks of L1 and L0 differ on the m and n axes is not supported yet");

    /// Construct
    __forceinline__ [aicore]
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_SIZE * STAGES;
        // Init buffers
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
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    }

    /// Destructor
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

    /// Perform a block-scoped matrix multiply-accumulate
    __forceinline__ [aicore]
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        AscendC::GlobalTensor<ElementA> const &gmNextBlockA,
        AscendC::GlobalTensor<ElementB> const &gmNextBlockB,
        GemmCoord const &actualShape, GemmCoord const &actualShapeNext,
        bool isFirstBlock, bool hasNextBlock)
    {
        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        auto layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(L1TileShape::M, L1TileShape::K);
        auto layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(L1TileShape::K, L1TileShape::N);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mRound, nRound));

        uint32_t kTileCount = CeilDiv<L1TileShape::K>(actualShape.k());
        uint32_t kTileCountNext = CeilDiv<L1TileShape::K>(actualShapeNext.k());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        }
        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx();
        }
        uint32_t firstTileIdx = startTileIdx % kTileCount;
        uint32_t lastTileIdx = (startTileIdx + kTileCount - 1) % kTileCount;
        uint32_t kActual =
            (firstTileIdx < kTileCount - 1) ? L1TileShape::K : (actualShape.k() - firstTileIdx * L1TileShape::K);
        uint32_t firstTileIdxNext = startTileIdx % kTileCountNext;

        uint32_t mPartLoop = CeilDiv<L0TileShape::M>(mRound);
        uint32_t nPartLoop = CeilDiv<L0TileShape::N>(nRound);

        // k loop
        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; kLoopIdx++) {
            uint32_t shuffleKIdx = (startTileIdx + kLoopIdx) % kTileCount;
            // Load first matrix A tile in total kernel loop from GM to L1
            if (shuffleKIdx == firstTileIdx && isFirstBlock) {
                MatrixCoord gmTileAOffset{0, shuffleKIdx * L1TileShape::K};
                MatrixCoord gmTileBOffset{shuffleKIdx * L1TileShape::K, 0};
                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
                // Load first matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
                copyGmToL1A(l1ATensorList[l1ListId], gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);

                // Load first matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
                copyGmToL1B(l1BTensorList[l1ListId], gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
            }

            uint32_t l1ListIdNext = (l1ListId + 1 < STAGES) ? (l1ListId + 1) : 0;
            uint32_t kActualNext{0};

            // preload next tile from GM to L1
            if (shuffleKIdx != lastTileIdx) {
                uint32_t shuffleKIdxNext = (startTileIdx + kLoopIdx + 1) % kTileCount;
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (shuffleKIdxNext < kTileCount - 1) ?
                    L1TileShape::K : (actualShape.k() - shuffleKIdxNext * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, shuffleKIdxNext * L1TileShape::K};
                MatrixCoord gmTileBOffset{shuffleKIdxNext * L1TileShape::K, 0};
                auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];

                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShape.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }
            if (shuffleKIdx == lastTileIdx && hasNextBlock) {
                // Get L1 tensor for next stage
                auto l1ATensor = l1ATensorList[l1ListIdNext];
                auto l1BTensor = l1BTensorList[l1ListIdNext];
                // Get GM tensor for next stage
                kActualNext = (firstTileIdxNext < kTileCountNext - 1) ?
                    L1TileShape::K : (actualShapeNext.k() - firstTileIdxNext * L1TileShape::K);
                MatrixCoord gmTileAOffset{0, firstTileIdxNext * L1TileShape::K};
                MatrixCoord gmTileBOffset{firstTileIdxNext * L1TileShape::K, 0};
                auto gmTileA = gmNextBlockA[layoutA.GetOffset(gmTileAOffset)];
                auto gmTileB = gmNextBlockB[layoutB.GetOffset(gmTileBOffset)];
                // load next matrix A tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListIdNext]);
                auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShapeNext.m(), kActualNext));
                copyGmToL1A(l1ATensor, gmTileA, layoutAInL1, layoutTileA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListIdNext]);

                // load next matrix B tile from GM to L1
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListIdNext]);
                auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActualNext, actualShapeNext.n()));
                copyGmToL1B(l1BTensor, gmTileB, layoutBInL1, layoutTileB);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListIdNext]);
            }

            // Get L1 tensor for current stage
            auto l1ATensor = l1ATensorList[l1ListId];
            auto l1BTensor = l1BTensorList[l1ListId];

            // Get the loop nums on L0
            uint32_t kPartLoop = CeilDiv<L0TileShape::K>(kActual);

            uint32_t l0ABufId = 0;
            uint32_t l0BBufId = 0;

            for (int mPartIdx = 0; mPartIdx < mPartLoop; mPartIdx++) {
                uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                    L0TileShape::M : (mRound - mPartIdx * L0TileShape::M);

                for (int kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                        L0TileShape::K : (kActual - kPartIdx * L0TileShape::K);

                    // Locate the current tile on L0A
                    auto l0ATile = l0ATensorList[l0ABufId];
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                    // Locate the current tile of matrix A on L1
                    MatrixCoord l1AOffset{mPartIdx * L0TileShape::M, kPartIdx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1AOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    if ((mPartIdx == 0) && (kPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
                    }

                    copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, layoutAInL1);

                    if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
                    }

                    for (int nPartIdx = 0; nPartIdx < nPartLoop; nPartIdx++) {
                        uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                            L0TileShape::N : (nRound - nPartIdx * L0TileShape::N);

                        // Locate the current tile on L0B
                        auto l0BTile = l0BTensorList[l0BBufId];
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                        // Locate the current tile of matrix B on L1
                        MatrixCoord l1BOffset{kPartIdx * L0TileShape::K, nPartIdx * L0TileShape::N};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BOffset)];

                        // Wait for mmad finished
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);
                        // If the current tile is the first one on the k&n axis, wait for loading matrix B from GM to L1
                        if ((kPartIdx == 0) && (nPartIdx == 0)) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);
                        }

                        // Load current tile from L1 to L0B
                        copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, layoutBInL1);

                        // If the current tile is the last one on the k&n axis, notify to load matrix B from GM to L1
                        if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
                        }
                        // Notify to do mmad
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // Locate the current tile on L0C
                        MatrixCoord l0COffset{mPartIdx * L0TileShape::M, nPartIdx * L0TileShape::N};
                        auto l0CTile = l0CTensor[layoutInL0C.GetOffset(l0COffset)];

                        // Compute the matrix multiplication on L0A and L0B and write the result to the accumulator
                        // Wait for loading L0B
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // If the current tile is the first tile on the k axis, the accumulator needs to be reset to 0
                        bool initC = ((kLoopIdx == 0) && (kPartIdx == 0));
                        // If the unit flag is enabled, the unit flag is set according to the calculation progress
                        uint8_t unitFlag = 0b00;
                        if constexpr (ENABLE_UNIT_FLAG) {
                            if ((kLoopIdx == kTileCount - 1) && (mPartIdx == mPartLoop - 1) &&
                                (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                                unitFlag = 0b11;
                            } else {
                                unitFlag = 0b10;
                            }
                        }
                        // Perform calculation operations
                        tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                        // Notify to move the next L0B tile
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BBufId]);

                        l0BBufId = (l0BBufId + 1 < STAGES) ? (l0BBufId + 1) : 0;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0ABufId]);
                    l0ABufId = (l0ABufId + 1 < STAGES) ? (l0ABufId + 1) : 0;
                }
            }

            l1ListId = l1ListIdNext;
            kActual = kActualNext;
        }

        // copy block out
        LayoutC layoutBlock = layoutC.GetTileLayout(actualShape.GetCoordMN());

        if constexpr (!ENABLE_UNIT_FLAG) {
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
            copyL0CToGm(gmBlockC, l0CTensor, layoutBlock, layoutInL0C);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        } else {
            copyL0CToGm(gmBlockC, l0CTensor, layoutBlock, layoutInL0C, 0b11);
        }
    }

protected:
    /// Data members
    AscendC::LocalTensor<ElementA> l1ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensorList[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensorList[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor;

    int32_t l1AEventList[STAGES];
    int32_t l1BEventList[STAGES];
    int32_t l0AEventList[STAGES];
    int32_t l0BEventList[STAGES];

    uint32_t l1ListId{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

} // namespace Catlass::Gemm::Block

// =================================================================================================
// Catlass GEMM - Kernel Level Operations
// =================================================================================================

namespace Catlass::Gemm::Kernel {
template<
    class ArchTag_,
    class Element_,
    class LayoutIn_,
    class LayoutOut_,
    uint32_t COMPUTE_LENGTH
>
struct PaddingMatrixBlockND {
public:
    using ArchTag = ArchTag_;
    using Element = Element_;
    using LayoutIn = LayoutIn_;
    using LayoutOut = LayoutOut_;
    using ComputeLayout = Catlass::layout::RowMajor;
    using ComputeLayoutDst = Catlass::layout::PaddingRowMajor;
    using CopyGm2Ub = Catlass::Epilogue::Tile::CopyGm2Ub<
        ArchTag, Gemm::GemmType<Element, ComputeLayout>>;
    using CopyUb2Gm = Catlass::Epilogue::Tile::CopyUb2Gm<
        ArchTag, Gemm::GemmType<Element, ComputeLayout>>;

    CopyGm2Ub copyGm2Ub;
    CopyUb2Gm copyUb2Gm;

    __forceinline__ [aicore]
    PaddingMatrixBlockND(Arch::Resource<ArchTag> &resource)
    {
        int64_t bufferOffset = 0;
        for (uint32_t i = 0; i < BUFFER_NUM; i++) {
            inputBuffer[i] = resource.ubBuf.template GetBufferByByte<Element>(bufferOffset * sizeof(Element));
            bufferOffset += COMPUTE_LENGTH;
        }
    }

    __forceinline__ [aicore]
    ComputeLayout GetPaddingComputeLayout(layout::RowMajor const &layout)
    {
        return ComputeLayout(layout.shape(0), layout.shape(1), layout.stride(0));
    }

    __forceinline__ [aicore]
    ComputeLayout GetPaddingComputeLayout(layout::ColumnMajor const &layout)
    {
        return ComputeLayout(layout.shape(1), layout.shape(0), layout.stride(1));
    }

    __forceinline__ [aicore]
    ComputeLayoutDst GetPaddingComputeLayout(layout::PaddingRowMajor const &layout)
    {
        return ComputeLayoutDst(
            layout.shape(0) * layout.shape(1),
            layout.shape(2) * layout.shape(3),
            layout.shape(0),
            layout.shape(2)
        );
    }

    __forceinline__ [aicore]
    ComputeLayoutDst GetPaddingComputeLayout(layout::PaddingColumnMajor const &layout)
    {
        return ComputeLayoutDst(
            layout.shape(2) * layout.shape(3),
            layout.shape(0) * layout.shape(1),
            layout.shape(2),
            layout.shape(0)
        );
    }

    __forceinline__ [aicore]
    void operator()(AscendC::GlobalTensor<Element> const &dst,
                    AscendC::GlobalTensor<Element> const &src,
                    LayoutOut layoutDst, LayoutIn layoutSrc)
    {
        auto computeLayoutSrc = GetPaddingComputeLayout(layoutSrc);
        auto computeLayoutDst = GetPaddingComputeLayout(layoutDst);

        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        uint32_t aivId = AscendC::GetBlockIdx();

        // Each line is a tile.
        uint32_t tilesNum = computeLayoutSrc.shape(0);
        uint32_t tileLen = computeLayoutSrc.shape(1);
        uint32_t roundTileLen = RoundUp<BYTE_PER_BLK / sizeof(Element)>(computeLayoutSrc.shape(1));

        uint32_t tilesPerAiv = tilesNum / aivNum;
        uint32_t tileRemain = tilesNum % aivNum;
        if (aivId < tileRemain) {
            tilesPerAiv++;
        }
        uint32_t mIdx = aivId * tilesPerAiv;
        if (aivId >= tileRemain) {
            mIdx += tileRemain;
        }
        MatrixCoord blockOffset(mIdx, 0);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
        uint32_t coreLoops{ 0 };
        if (roundTileLen > COMPUTE_LENGTH) {
            // Handle the same tile on multiple loops.
            uint32_t loopsPerTile = (tileLen + COMPUTE_LENGTH - 1) / COMPUTE_LENGTH;
            coreLoops = tilesPerAiv * loopsPerTile;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx / loopsPerTile;
                uint32_t inTileLoopIdx = loopIdx % loopsPerTile;
                MatrixCoord loopOffset(tileIdx, inTileLoopIdx * COMPUTE_LENGTH);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + loopOffset);
                uint32_t actualDataNum = COMPUTE_LENGTH;
                if (tileLen - inTileLoopIdx * COMPUTE_LENGTH < COMPUTE_LENGTH) {
                    actualDataNum = tileLen - inTileLoopIdx * COMPUTE_LENGTH;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(1, actualDataNum));
                ComputeLayout ubLayout = ComputeLayout{1, actualDataNum};
                ComputeLayout dstLayout = ComputeLayout(CeilDiv(actualDataNum, computeLayoutDst.shape(2)),
                    computeLayoutDst.shape(2), computeLayoutDst.stride(3));

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + loopOffset);
                copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex], dstLayout, ubLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        } else {
            // Handle multiple tile each loop.
            uint32_t tilesPerLoop = COMPUTE_LENGTH / roundTileLen;
            coreLoops = (tilesPerAiv + tilesPerLoop - 1) / tilesPerLoop;
            for (uint32_t loopIdx = 0; loopIdx < coreLoops; ++loopIdx) {
                uint32_t tileIdx = loopIdx * tilesPerLoop;
                MatrixCoord tileOffset(tileIdx, 0);
                uint64_t gmSrcOffset = computeLayoutSrc.GetOffset(blockOffset + tileOffset);
                uint32_t actualTilesNum = tilesPerLoop;
                if (tilesPerAiv - tileIdx < tilesPerLoop) {
                    actualTilesNum = tilesPerAiv - tileIdx;
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);
                ComputeLayout srcLayout = computeLayoutSrc.GetTileLayout(MatrixCoord(actualTilesNum, tileLen));
                ComputeLayout ubLayout = ComputeLayout{actualTilesNum, tileLen, roundTileLen};
                ComputeLayout dstLayout = ComputeLayout{CeilDiv(tileLen, computeLayoutDst.shape(2)),
                    computeLayoutDst.shape(2), computeLayoutDst.stride(3)};

                copyGm2Ub(inputBuffer[bufferIndex], src[gmSrcOffset], ubLayout, srcLayout);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIds[bufferIndex]);

                for (uint32_t i = 0; i < actualTilesNum; ++i) {
                    uint64_t gmDstOffset = computeLayoutDst.GetOffset(blockOffset + tileOffset + MatrixCoord(i, 0));
                    uint64_t ubOffset = ubLayout.GetOffset(MatrixCoord(i, 0));
                    copyUb2Gm(dst[gmDstOffset], inputBuffer[bufferIndex][ubOffset], dstLayout, ubLayout);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[bufferIndex]);

                bufferIndex = (bufferIndex + 1) % BUFFER_NUM;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[0]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIds[1]);
    }

    __forceinline__ [aicore]
    ~PaddingMatrixBlockND() {}
private:
    static const uint32_t BUFFER_NUM = 2;
    AscendC::LocalTensor<Element> inputBuffer[BUFFER_NUM];
    AscendC::TEventID eventIds[BUFFER_NUM] = {EVENT_ID0, EVENT_ID1};
    uint32_t bufferIndex{ 0 };
    static_assert(BUFFER_NUM * COMPUTE_LENGTH * sizeof(Element) <= ArchTag::UB_SIZE, "Excedding the UB space!");
};


template <
    class PrologueA,
    class PrologueB,
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class OptimizedMatmul {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template<class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template<>
    struct LayoutHelper<void> {
        using type = void;
    };

    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, LayoutWA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, LayoutWB, typename LayoutHelper<PrologueB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrC;
        LayoutC layoutC;
        GM_ADDR ptrWA;
        LayoutWA layoutWA;
        GM_ADDR ptrWB;
        LayoutWB layoutWB;

        // Methods
        __forceinline__ [aicore]
        Params() {}

        __forceinline__ [aicore]
        Params(GemmCoord const &problemShape_,
               GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_, GM_ADDR ptrC_, LayoutC layoutC_,
               GM_ADDR ptrWA_, LayoutWA layoutWA_, GM_ADDR ptrWB_, LayoutWB layoutWB_)
            : problemShape(problemShape_), ptrA(ptrA_), layoutA(layoutA_), ptrB(ptrB_), layoutB(layoutB_),
              ptrC(ptrC_), layoutC(layoutC_), ptrWA(ptrWA_), layoutWA(layoutWA_), ptrWB(ptrWB_), layoutWB(layoutWB_) {}
    };

    struct Arguments {
        GemmCoord problemShape;
        uint32_t align;
        size_t elementSize;
        LayoutWA layoutWA;
        LayoutWB layoutWB;
        GM_ADDR ptrA;
        GM_ADDR ptrB;
        GM_ADDR ptrC;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static bool IfNeedPadding(layout::RowMajor layout, uint32_t align)
    {
        // prevent division by zero
        if (align == 0) {
            return false;
        }
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(0) < 65536) {
            return layout.stride(0) % align != 0;
        } else {
            return true;
        }
    }

    static bool IfNeedPadding(layout::ColumnMajor layout, uint32_t align)
    {
        // prevent division by zero
        if (align == 0) {
            return false;
        }
        // If the stride is greater than 65536, padding is required to reduce the stride.
        if (layout.stride(1) < 65536) {
            return layout.stride(1) % align != 0;
        } else {
            return true;
        }
    }

    template<class Layout>
    static size_t GetWorkspaceLen(Layout layout, size_t blockRows, size_t blockCols)
    {
        return RoundUp(static_cast<size_t>(layout.shape(0)), blockRows) *
                RoundUp(static_cast<size_t>(layout.shape(1)), blockRows);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        size_t workspaceSize = 0;
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        if (IfNeedPadding(layoutA, args.align)) {
            workspaceSize +=
                GetWorkspaceLen(layoutA, args.layoutWA.shape(0), args.layoutWA.shape(2)) * args.elementSize;
        }
        if (IfNeedPadding(layoutB, args.align)) {
            workspaceSize +=
                GetWorkspaceLen(layoutB, args.layoutWB.shape(0), args.layoutWB.shape(2)) * args.elementSize;
        }
        return workspaceSize;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        using LayoutPaddingA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        using LayoutPaddingB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
        LayoutA layoutA{args.problemShape.m(), args.problemShape.k()};
        LayoutB layoutB{args.problemShape.k(), args.problemShape.n()};
        LayoutC layoutC{args.problemShape.m(), args.problemShape.n()};
        bool isPaddingA = IfNeedPadding(layoutA, args.align);
        bool isPaddingB = IfNeedPadding(layoutB, args.align);

        uint8_t *gmWA = nullptr;
        uint8_t *gmWB = nullptr;
        size_t sizeWA = 0;

        if (isPaddingA) {
            gmWA = workspace;
            sizeWA = GetWorkspaceLen(layoutA, args.layoutWA.shape(0), args.layoutWA.shape(2)) * args.elementSize;
        } else {
            gmWA = args.ptrA;
        }
        if (isPaddingB) {
            gmWB = workspace + sizeWA;
        } else {
            gmWB = args.ptrB;
        }
        Params params{args.problemShape,
            args.ptrA,
            layoutA,
            args.ptrB,
            layoutB,
            args.ptrC,
            layoutC,
            gmWA,
            args.layoutWA,
            gmWB,
            args.layoutWB};
        return params;
    }

    // Methods
    __forceinline__ [aicore]
    OptimizedMatmul() {}

    template <int32_t CORE_TYPE = g_coreType>
    __forceinline__ [aicore]
    void operator()(Params const &params);

    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIV>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA>) {
            AscendC::GlobalTensor<ElementA> gmA;
            AscendC::GlobalTensor<ElementA> gmWA;
            gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
            gmWA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrWA));
            PrologueA prologueA(resource);
            prologueA(gmWA, gmA, params.layoutWA, params.layoutA);
        }

        if constexpr (!std::is_void_v<PrologueB>) {
            AscendC::GlobalTensor<ElementB> gmB;
            AscendC::GlobalTensor<ElementB> gmWB;
            gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
            gmWB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrWB));
            PrologueB prologueB(resource);
            prologueB(gmWB, gmB, params.layoutWB, params.layoutB);
            // 0x0 synchronization control between AI Core
        }
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishPadding);
        }
    }

    /// Executes matmul
    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIC>(Params const &params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        BlockScheduler matmulBlockScheduler(params.problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA *)params.ptrWA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB *)params.ptrWB);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC *)params.ptrC);

        BlockMmad blockMmad(resource);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            // Compute block location
            GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockIdxCoord);

            // Compute initial location in logical coordinates
            MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockIdxCoord.k() * L1TileShape::K, blockIdxCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.n() * L1TileShape::N};
            int64_t gmOffsetA = params.layoutWA.GetOffset(offsetA);
            int64_t gmOffsetB = params.layoutWB.GetOffset(offsetB);
            int64_t gmOffsetC = params.layoutC.GetOffset(offsetC);

            bool isFirstBlock = (loopIdx == AscendC::GetBlockIdx());
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextActualBlockShape;
            if (loopIdx + AscendC::GetBlockNum() < coreLoops) {
                hasNextBlock = true;
                nextBlockIdCoord = matmulBlockScheduler.GetBlockCoord(loopIdx + AscendC::GetBlockNum());
                nextActualBlockShape = matmulBlockScheduler.GetActualBlockShape(nextBlockIdCoord);
            }
            MatrixCoord offsetNextA{nextBlockIdCoord.m() * L1TileShape::M, nextBlockIdCoord.k() * L1TileShape::K};
            MatrixCoord offsetNextB{nextBlockIdCoord.k() * L1TileShape::K, nextBlockIdCoord.n() * L1TileShape::N};
            int64_t gmOffsetNextA = params.layoutWA.GetOffset(offsetNextA);
            int64_t gmOffsetNextB = params.layoutWB.GetOffset(offsetNextB);

            // Compute block-scoped matrix multiply-add
            blockMmad(
                gmA[gmOffsetA], params.layoutWA,
                gmB[gmOffsetB], params.layoutWB,
                gmC[gmOffsetC], params.layoutC,
                gmA[gmOffsetNextA], gmB[gmOffsetNextB],
                actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

// =================================================================================================
// Kernel Entry Point
// =================================================================================================

using namespace Catlass;
using namespace matmul;


template <
    /// Tag indicating architecture
    class ArchTag,
    /// GemmType for A matrix operand
    class AType,
    /// GemmType type for B matrix operand
    class BType,
    /// GemmType type for C matrix operand
    class CType,
    /// GemmType type for Bias operand
    class BiasType = void
>
struct TileCopyOpt : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType, BType, CType, BiasType>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    // When matrix A is row-major, if the number of rows in matrix A is less than 16, 
    // using the CopyGmToL1IntervalDataCopy method can improve the transfer efficiency.
    // The situation is similar for matrix B. If the above conditions are met, 
    // please uncomment the following and comment out the original matrix A transfer method

    // using CopyGmToL1A = Gemm::Tile::CopyGmToL1IntervalDataCopy<ArchTag, AType>;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm; 
    using BiasTypeSelector = typename Base::BiasTypeSelector; 
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};

using ArchTag = Arch::AtlasA2;
constexpr bool ENABLE_UNIT_FLAG = true;
constexpr bool ENABLE_SHUFFLE_K = true;
using ElementA = half;
using ElementB = half;
using ElementC = half;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;
using LayoutPaddingA = std::conditional_t<std::is_same_v<LayoutA, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
using LayoutPaddingB = std::conditional_t<std::is_same_v<LayoutB, layout::RowMajor>,
            layout::PaddingRowMajor, layout::PaddingColumnMajor>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using ATypePadding = Gemm::GemmType<ElementA, LayoutPaddingA>;
using BTypePadding = Gemm::GemmType<ElementB, LayoutPaddingB>;
using DispatchPolicy = Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>;

// if LayoutA and LayoutB is both ColumnMajor,
// L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
    std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;
using L0TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
    std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 64>, GemmShape<128, 256, 64>>;
using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using BlockScheduler31 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
using BlockEpilogue = void;

extern "C" __global__ __aicore__ void optimized_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t m = tiling_data.m;
    uint32_t k = tiling_data.k;
    uint32_t n = tiling_data.n;
    uint32_t paddingASize = tiling_data.paddingASize;
    bool isNeedPaddingA = paddingASize > 0;
    uint32_t paddingBSize = tiling_data.paddingBSize;
    bool isNeedPaddingB = paddingBSize > 0;

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    static const uint32_t COMPUTE_LENGTH_A = 96 * 1024 / sizeof(ElementA);
    using GlobalPaddingA = Gemm::Kernel::PaddingMatrixBlockND<
        ArchTag, ElementA, LayoutA, LayoutPaddingA, COMPUTE_LENGTH_A>;
    static const uint32_t COMPUTE_LENGTH_B = 96 * 1024 / sizeof(ElementB);
    using GlobalPaddingB = Gemm::Kernel::PaddingMatrixBlockND<
        ArchTag, ElementB, LayoutB, LayoutPaddingB, COMPUTE_LENGTH_B>;

    // if LayoutA and LayoutB is both ColumnMajor,
    // L1TileShape using GemmShape<256, 128, 256> can achieve better performance.
    using L1TileShape = std::conditional_t<std::is_same_v<LayoutA, layout::ColumnMajor> &&
        std::is_same_v<LayoutB, layout::ColumnMajor>, GemmShape<256, 128, 256>, GemmShape<128, 256, 256>>;

    if (m > n) {
        if (isNeedPaddingA && isNeedPaddingB) {
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            using TileCopy = TileCopyOpt<ArchTag, ATypePadding, BTypePadding, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BTypePadding, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            workspace, layoutWA, workspace + paddingASize, layoutWB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else if (isNeedPaddingA) {
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            using TileCopy = TileCopyOpt<ArchTag, ATypePadding, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            workspace, layoutWA, b, layoutB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else if (isNeedPaddingB) {
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            using TileCopy = TileCopyOpt<ArchTag, AType, BTypePadding, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BTypePadding, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            a, layoutA, workspace, layoutWB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else {
            using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler30>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            a, layoutA, b, layoutB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        }
    } else {
        if (isNeedPaddingA && isNeedPaddingB) {
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            using TileCopy = TileCopyOpt<ArchTag, ATypePadding, BTypePadding, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BTypePadding, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            workspace, layoutWA, workspace + paddingASize, layoutWB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else if (isNeedPaddingA) {
            LayoutPaddingA layoutWA = LayoutPaddingA(
                layoutA.shape(0), layoutA.shape(1), L1TileShape::M, L1TileShape::K);
            using TileCopy = TileCopyOpt<ArchTag, ATypePadding, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, ATypePadding, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                GlobalPaddingA, void, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            workspace, layoutWA, b, layoutB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else if (isNeedPaddingB) {
            LayoutPaddingB layoutWB = LayoutPaddingB(
                layoutB.shape(0), layoutB.shape(1), L1TileShape::K, L1TileShape::N);
            using TileCopy = TileCopyOpt<ArchTag, AType, BTypePadding, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BTypePadding, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, GlobalPaddingB, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            a, layoutA, workspace, layoutWB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        } else {
            using TileCopy = TileCopyOpt<ArchTag, AType, BType, CType>;
            using BlockMmadOpt = Gemm::Block::BlockMmad<
                DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopy>;
            using MatmulKernel = Gemm::Kernel::OptimizedMatmul<
                void, void, BlockMmadOpt, BlockEpilogue, BlockScheduler31>;
            typename MatmulKernel::Params params{problemShape, a, layoutA, b, layoutB, c, layoutC,
            a, layoutA, b, layoutB};

            // call a kernel
            MatmulKernel matmul;
            matmul(params);
        }
    }
}