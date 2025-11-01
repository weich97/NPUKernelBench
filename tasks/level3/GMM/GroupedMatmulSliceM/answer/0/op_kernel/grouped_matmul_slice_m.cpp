#include <tuple>
#include <kernel_operator.h>
#include <type_traits>
#include <cstdlib>
#include "lib/matmul_intf.h"

#define __TLA_REQUIRES(...)   typename std::enable_if<(__VA_ARGS__)>::type* = nullptr

namespace tla {

template <class T>
struct remove_cvref {
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

} // end namespace tla

template <bool VALUE, class... Args>
constexpr bool DEPENDENT_BOOL_VALUE = VALUE;

template <class... Args>
constexpr bool DEPENDENT_FALSE = DEPENDENT_BOOL_VALUE<false, Args...>;

/// @brief Callback is an alternative to std::function<void(void)>, providing a general carrier
/// of callable structure with no parameters and no return value. Compared with function pointers
/// of type void (*)(), Callback can carry lambda expressions with captures, and does not need to
/// pay attention to the captured content. It should be noted that Callback itself does not store
/// the callable structure it carries like std::function<void(void)>, so it is necessary to ensure
/// that it is used within the life cycle of the callable structure.
struct Callback {
    void const *func{nullptr};
    void (*caller)(void const *){nullptr};

    Callback() = default;

    __forceinline__ [aicore]
    void operator()() const
    {
        if (func) {
            caller(func);
        }
    }

    __forceinline__ [aicore]
    operator bool() const
    {
        return func != nullptr;
    }
};

template <typename Func>
__forceinline__ [aicore]
void FuncWrapper(void const *func)
{
    (*static_cast<Func const *>(func))();
}

// Use this to make a callback
template <typename Func>
__forceinline__ [aicore]
Callback MakeCallback(Func *func)
{
    Callback callback;
    callback.func = func;
    callback.caller = &FuncWrapper<Func>;
    return callback;
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

template <uint32_t ALIGN, typename T>
__forceinline__ [aicore]
constexpr T RoundUp(const T &val)
{
    static_assert(ALIGN != 0, "ALIGN must not be 0");
    return (val + ALIGN - 1) / ALIGN * ALIGN;
}

namespace Catlass {

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;

constexpr uint32_t STRIDE_LIMIT = 65536;

}  // namespace Catlass

namespace Catlass {

/// Statically-sized array specifying Coords within a tensor
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
    // Indices
    Index idx[RANK];
};

// Helper to make a 1-element coordinate
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

/// Helper to make a 3-element coordinate
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

}  // namespace Catlass

namespace Catlass {

/// Shape of a matrix multiply-add operation
template <
    /// Rows of matrix product
    uint32_t M_ = 1,
    /// Columns of matrix product
    uint32_t N_ = 1,
    /// Inner dimension of matrix product
    uint32_t K_ = 1
>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;

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

    __forceinline__ [aicore]
    static Coord<2> ToCoordMN()
    {
        return MakeCoord(M, N);
    }
};

/// GemmCoord is a structure derived from Coord<3> that specifies a location within the
/// coordinate space of a Gemm problem.
struct GemmCoord : public Coord<3, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=3
    using Base = Coord<3, Index>;

    /// Gemm M dimension - rows of the output C matrix
    static constexpr int M_INDEX = 0;

    /// Gemm N dimension - columns of the output C matrix
    static constexpr int N_INDEX = 1;

    /// Gemm K dimension - inner dimension of the Gemm problem
    static constexpr int K_INDEX = 2;

    /// Default ctor
    __forceinline__ [aicore]
    GemmCoord() {}

    /// Constructs from Coord<3> and a batch
    __forceinline__ [aicore]
    GemmCoord(Coord<3, Index> const &coord) : Base(coord) {}

    /// Helper to construct from a K, N, M, batch variables
    __forceinline__ [aicore]
    GemmCoord(Index m, Index n, Index k) : Base(MakeCoord(m, n, k)) {}

    /// Returns the Gemm M coordinate
    __forceinline__ [aicore]
    Index const &m() const
    {
        return this->At(M_INDEX);
    }

    /// Returns the Gemm N coordinate
    __forceinline__ [aicore]
    Index const &n() const
    {
        return this->At(N_INDEX);
    }

    /// Returns the Gemm K coordinate
    __forceinline__ [aicore]
    Index const &k() const
    {
        return this->At(K_INDEX);
    }

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

} // namespace Catlass

namespace Catlass {
/// MatrixCoord wraps Coord<2, uint32_t> to provide a helper for accessing named dimensions. Classes
/// expecting a coordinate in the rank=2 index space of a matrix should use MatrixCoord.
struct MatrixCoord : public Coord<2, uint32_t> {
    /// Integer-valued index
    using Index = uint32_t;

    /// Base type is a Coord of rank=2
    using Base = Coord<2, Index>;

    /// LongIndex type
    using LongIndex = typename Base::LongIndex;

    /// Rows dimension
    static constexpr uint32_t ROW_INDEX = 0;

    /// Columns dimension
    static constexpr uint32_t COLUMN_INDEX = 1;

    /// Default ctor
    __forceinline__ [aicore]
    MatrixCoord() {}

    /// Constructs from Coord<2>
    __forceinline__ [aicore]
    MatrixCoord(Coord<2, Index> const &coord) : Base(coord) {}

    /// Helper to construct from a row and column
    __forceinline__ [aicore]
    MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column)) {}

    /// Returns the row of the coordinate
    __forceinline__ [aicore]
    Index const &row() const { return this->At(ROW_INDEX); }

    /// Returns the column of the coordinate
    __forceinline__ [aicore]
    Index const &column() const { return this->At(COLUMN_INDEX); }
};

} // namespace Catlass

namespace Catlass::layout {

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

}  // namespace Catlass::layout

namespace Catlass::Gemm {

template <class Element_, class Layout_, AscendC::TPosition POSITION_ = AscendC::TPosition::GM>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr AscendC::TPosition POSITION = POSITION_;
};

} // namespace Catlass::Gemm

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

} // namespace Catlass::Arch

namespace Catlass::Gemm {

template <bool ASYNC_ = false>
struct MmadAtlasA2Base {
    using ArchTag = Arch::AtlasA2;
    static constexpr uint32_t ASYNC = ASYNC_;
};

using MmadAtlasA2Async = MmadAtlasA2Base<true>;

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA2PreloadAsync : public MmadAtlasA2Async {
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

}  // namespace Catlass::Gemm

namespace Catlass::Arch {

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

template <
    class ArchTag,
    AscendC::TPosition Position
>
struct LocalTensorBuffer {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported local tensor buffer, can not find the specialization.");
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A1;
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
    static constexpr AscendC::TPosition Position = AscendC::TPosition::A2;
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
    static constexpr AscendC::TPosition Position = AscendC::TPosition::B2;
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
    using ArchTag = Arch::AtlasA2;
    static constexpr AscendC::TPosition Position = AscendC::TPosition::C2;
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::C2> tbufC2;
        GetTPipePtr()->InitBuffer(tbufC2, ArchTag::BIAS_SIZE);
        tensor = tbufC2.Get<uint8_t>();
    }
};

template <class ArchTag>
struct LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> : LocalTensorBufferBase {
public:
    static constexpr AscendC::TPosition Position = AscendC::TPosition::CO1;
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
    static constexpr AscendC::TPosition Position = AscendC::TPosition::VECCALC;
    __forceinline__ [aicore]
    LocalTensorBuffer()
    {
        AscendC::TBuf<AscendC::TPosition::VECCALC> tbufVECCALC;
        GetTPipePtr()->InitBuffer(tbufVECCALC, ArchTag::UB_SIZE);
        tensor = tbufVECCALC.Get<uint8_t>();
    }
};

}  // namespace Catlass::Arch

namespace Catlass::Gemm::Block {

/// Block swizzling function for Gemms
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    /// Data members
    GemmCoord problemShape;
    MatrixCoord tileMN;
    MatrixCoord loopsMN;

    /// Methods
    __forceinline__ [aicore]
    GemmIdentityBlockSwizzle() {}

    __forceinline__ [aicore]
    void Update(GemmCoord const &problemShape_, MatrixCoord const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
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
        } else if constexpr (SwizzleDirection == 1) { // Nz
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

namespace Catlass::Arch {

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

namespace Catlass::Gemm::helper {

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

template<class Element>
struct L1AlignHelper<Element, layout::ColumnMajor> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = C0_NUM_PER_FRACTAL;
};

template<class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>,
        "Unsupported element accumulator selector, can not find the specialization.");
};

template<>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template<class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1ATypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1AType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
};

template<class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>,
        "Unsupported layout selector, can not find the specialization.");
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::RowMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>;
};

template<class Element>
struct L1BTypeSelector<Gemm::GemmType<Element, layout::ColumnMajor>> {
    using L1BType = Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>;
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

} // namespace Catlass::Gemm::helper

namespace Catlass::Gemm::Tile {
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

}  // namespace Catlass::Gemm::Tile

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

template <class Element>
struct CopyGmToL1<Arch::AtlasA2, Gemm::GemmType<Element, layout::ColumnMajor>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::ColumnMajor;
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
        intriParams.dValue = layoutSrc.shape(0);
        intriParams.srcNdMatrixStride = 0;
        intriParams.dstNzC0Stride = layoutDst.stride(1) / ELE_NUM_PER_C0;
        intriParams.dstNzMatrixStride = 0;

        if (layoutSrc.stride(1) < STRIDE_LIMIT) {
            intriParams.nValue = layoutSrc.shape(1);
            intriParams.srcDValue = layoutSrc.stride(1);
            intriParams.dstNzNStride = layoutDst.stride(2) / ELE_NUM_PER_C0;
            AscendC::DataCopy(dstTensor, srcTensor, intriParams);
        } else {
            intriParams.nValue = 1;
            intriParams.srcDValue = 0;
            intriParams.dstNzNStride = 0;
            for (uint32_t i = 0; i < layoutSrc.shape(1); i++) {
                AscendC::DataCopy(dstTensor[i * ELE_NUM_PER_C0], srcTensor[i * layoutSrc.stride(1)], intriParams);
            }
        }
    }
};

} // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::Tile {

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
        loadDataParams.srcStride = layoutSrc.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element));
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element)) - 1;
        loadDataParams.ifTranspose = false;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0A<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::zZ;
    using LayoutSrc = layout::nZ;

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
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element));
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element)) - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < CeilDiv<BYTE_PER_C0 / sizeof(Element)>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

} // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::Tile {

template <
    class ArchTag,
    class L1Type,
    class L0Type = void
>
struct CopyL1ToL0B {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0, can not find the specialization.");
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::zN, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::zN;

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
        loadDataParams.repeatTimes = static_cast<uint16_t>(CeilDiv<BYTE_PER_C0 / sizeof(Element)>(layoutDst.orgShape(1)));
        loadDataParams.srcStride = layoutSrc.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element));
        loadDataParams.sid = 0;
        loadDataParams.dstGap = layoutDst.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element)) - 1;
        loadDataParams.ifTranspose = true;
        loadDataParams.addrMode = 0;
        for (uint32_t i = 0; i < CeilDiv<C0_NUM_PER_FRACTAL>(layoutDst.orgShape(0)); i++) {
            AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
        }
    }
};

template <class ArchTag, class Element>
struct CopyL1ToL0B<ArchTag, Gemm::GemmType<Element, layout::nZ, AscendC::TPosition::A1>> {
    using LayoutDst = layout::nZ;
    using LayoutSrc = layout::nZ;

    __forceinline__ [aicore]
    CopyL1ToL0B() {};

    __forceinline__ [aicore]
    void operator()(
        AscendC::LocalTensor<Element> const &dstTensor,
        AscendC::LocalTensor<Element> const &srcTensor,
        LayoutDst const &layoutDst, LayoutSrc const &layoutSrc)
    {
        AscendC::LoadData2DParams loadDataParams;
        if (layoutSrc.shape(3) == layoutDst.shape(3)) {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(1) * layoutDst.shape(3));
            loadDataParams.srcStride = 1;
            loadDataParams.sid = 0;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;
            AscendC::LoadData(dstTensor, srcTensor, loadDataParams);
        } else {
            loadDataParams.startIndex = 0;
            loadDataParams.repeatTimes = static_cast<uint16_t>(layoutDst.shape(3));
            loadDataParams.srcStride = layoutSrc.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element));
            loadDataParams.sid = 0;
            loadDataParams.dstGap = layoutDst.stride(3) / (BYTE_PER_FRACTAL / sizeof(Element)) - 1;
            loadDataParams.ifTranspose = false;
            loadDataParams.addrMode = 0;
            for (uint32_t i = 0; i < layoutDst.shape(1); i++) {
                AscendC::LoadData(dstTensor[i * layoutDst.stride(1)], srcTensor[i * layoutSrc.stride(1)], loadDataParams);
            }
        }
    }
};

} // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::Tile {

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
};

} // namespace Catlass::Gemm::Tile

namespace Catlass::Gemm::Tile {

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
        void>; // Simplified, as Bias is not used
};

} // namespace Catlass::Gemm::Tile

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
    uint32_t PRELOAD_STAGES_,
    uint32_t L1_STAGES_,
    uint32_t L0A_STAGES_,
    uint32_t L0B_STAGES_,
    uint32_t L0C_STAGES_,
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
    MmadAtlasA2PreloadAsync<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    >,
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
    using DispatchPolicy = MmadAtlasA2PreloadAsync<
        PRELOAD_STAGES_,
        L1_STAGES_,
        L0A_STAGES_,
        L0B_STAGES_,
        L0C_STAGES_,
        ENABLE_UNIT_FLAG_,
        ENABLE_SHUFFLE_K_
    >;
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

    static constexpr uint32_t PRELOAD_STAGES = DispatchPolicy::PRELOAD_STAGES;
    static constexpr uint32_t L1_STAGES = DispatchPolicy::L1_STAGES;
    static constexpr uint32_t L0A_STAGES = DispatchPolicy::L0A_STAGES;
    static constexpr uint32_t L0B_STAGES = DispatchPolicy::L0B_STAGES;
    static constexpr uint32_t L0C_STAGES = DispatchPolicy::L0C_STAGES;
    static constexpr bool ENABLE_UNIT_FLAG = DispatchPolicy::ENABLE_UNIT_FLAG;
    static constexpr bool ENABLE_SHUFFLE_K = DispatchPolicy::ENABLE_SHUFFLE_K;

    static constexpr uint32_t L1A_TILE_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L1B_TILE_SIZE = L1TileShape::N * L1TileShape::K * sizeof(ElementB);
    static constexpr uint32_t L0A_TILE_SIZE = L0TileShape::M * L0TileShape::K * sizeof(ElementA);
    static constexpr uint32_t L0B_TILE_SIZE = L0TileShape::K * L0TileShape::N * sizeof(ElementB);
    static constexpr uint32_t L0C_TILE_SIZE = L1TileShape::M * L1TileShape::N * sizeof(ElementAccumulator);

    static constexpr auto L1A_LAYOUT = LayoutAInL1::template MakeLayout<ElementA>(
        L1TileShape::M, L1TileShape::K);
    static constexpr auto L1B_LAYOUT = LayoutBInL1::template MakeLayout<ElementB>(
        L1TileShape::K, L1TileShape::N);

    __forceinline__ [aicore]
    BlockMmad(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart = 0)
    {
        InitL1(resource, l1BufAddrStart);
        InitL0A(resource);
        InitL0B(resource);
        InitL0C(resource);
    }

    __forceinline__ [aicore]
    ~BlockMmad()
    {
        SynchronizeBlock();
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    __forceinline__ [aicore]
    void operator()(
        AscendC::GlobalTensor<ElementA> const &gmBlockA, LayoutA const &layoutA,
        AscendC::GlobalTensor<ElementB> const &gmBlockB, LayoutB const &layoutB,
        AscendC::GlobalTensor<ElementC> const &gmBlockC, LayoutC const &layoutC,
        GemmCoord const &actualShape, Callback &&callback = Callback{}
    )
    {
        uint32_t kTileCount = CeilDiv(actualShape.k(), (uint32_t)L1TileShape::K);

        uint32_t mRound = RoundUp<L1AAlignHelper::M_ALIGNED>(actualShape.m());
        uint32_t nRound = RoundUp<L1BAlignHelper::N_ALIGNED>(actualShape.n());

        uint32_t startTileIdx = 0;
        if constexpr (ENABLE_SHUFFLE_K) {
            startTileIdx = AscendC::GetBlockIdx() % kTileCount;
        }

        for (uint32_t kLoopIdx = 0; kLoopIdx < kTileCount; ++kLoopIdx) {
            uint32_t kTileIdx = (startTileIdx + kLoopIdx < kTileCount) ?
                (startTileIdx + kLoopIdx) : (startTileIdx + kLoopIdx - kTileCount);

            uint32_t kActual = (kTileIdx < kTileCount - 1) ?
                L1TileShape::K : (actualShape.k() - kTileIdx * L1TileShape::K);

            MatrixCoord gmTileAOffset{0, kTileIdx * L1TileShape::K};
            MatrixCoord gmTileBOffset{kTileIdx * L1TileShape::K, 0};
            auto gmTileA = gmBlockA[layoutA.GetOffset(gmTileAOffset)];
            auto gmTileB = gmBlockB[layoutB.GetOffset(gmTileBOffset)];
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[l1ListId]);
            auto layoutTileA = layoutA.GetTileLayout(MakeCoord(actualShape.m(), kActual));
            copyGmToL1A(l1ATensorList[l1ListId], gmTileA, L1A_LAYOUT, layoutTileA);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[l1ListId]);
            
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[l1ListId]);
            auto layoutTileB = layoutB.GetTileLayout(MakeCoord(kActual, actualShape.n()));
            copyGmToL1B(l1BTensorList[l1ListId], gmTileB, L1B_LAYOUT, layoutTileB);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[l1ListId]);

            if (preloadCount == PRELOAD_STAGES) {
                L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            }
            
            uint32_t preloadL1TileMmadParamsId = (l1TileMmadParamsId + preloadCount < PRELOAD_STAGES) ?
                (l1TileMmadParamsId + preloadCount) : (l1TileMmadParamsId + preloadCount - PRELOAD_STAGES);
            auto &l1TileMmadParams = l1TileMmadParamsList[preloadL1TileMmadParamsId];
            l1TileMmadParams.l1ListId = l1ListId;
            l1TileMmadParams.mRound = mRound;
            l1TileMmadParams.nRound = nRound;
            l1TileMmadParams.kActual = kActual;
            l1TileMmadParams.isKLoopFirst = (kLoopIdx == 0);
            l1TileMmadParams.isKLoopLast = (kLoopIdx == kTileCount - 1);
            if (kLoopIdx == kTileCount - 1) {
                l1TileMmadParams.gmBlockC = gmBlockC;
                l1TileMmadParams.layoutCInGm = layoutC.GetTileLayout(actualShape.GetCoordMN());
                l1TileMmadParams.callback = callback;
            }

            if (preloadCount < PRELOAD_STAGES) {
                ++preloadCount;
            } else {
                l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            }
            l1ListId = (l1ListId + 1 < L1_STAGES) ? (l1ListId + 1) : 0;
        }
    }

    __forceinline__ [aicore]
    void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }

private:
    struct L1TileMmadParams {
        uint32_t l1ListId;
        uint32_t mRound;
        uint32_t nRound;
        uint32_t kActual;
        bool isKLoopFirst;
        bool isKLoopLast;
        AscendC::GlobalTensor<ElementC> gmBlockC;
        LayoutC layoutCInGm;
        Callback callback;
        __forceinline__ [aicore] L1TileMmadParams() = default;
    };

    __forceinline__ [aicore]
    void InitL1(Arch::Resource<ArchTag> &resource, uint32_t l1BufAddrStart)
    {
        uint32_t l1AOffset = l1BufAddrStart;
        uint32_t l1BOffset = l1BufAddrStart + L1A_TILE_SIZE * L1_STAGES;
        for (uint32_t i = 0; i < L1_STAGES; ++i) {
            l1ATensorList[i] = resource.l1Buf.template GetBufferByByte<ElementA>(l1AOffset + L1A_TILE_SIZE * i);
            l1BTensorList[i] = resource.l1Buf.template GetBufferByByte<ElementB>(l1BOffset + L1B_TILE_SIZE * i);
            l1AEventList[i] = i;
            l1BEventList[i] = i + L1_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[i]);
        }
    }

    __forceinline__ [aicore] void InitL0A(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0A_STAGES; ++i) {
            l0ATensorList[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_TILE_SIZE * i);
            l0AEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[i]);
        }
    }

    __forceinline__ [aicore] void InitL0B(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0B_STAGES; ++i) {
            l0BTensorList[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_TILE_SIZE * i);
            l0BEventList[i] = i + L0A_STAGES;
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[i]);
        }
    }

    __forceinline__ [aicore] void InitL0C(Arch::Resource<ArchTag> &resource)
    {
        for (uint32_t i = 0; i < L0C_STAGES; ++i) {
            l0CTensorList[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_TILE_SIZE * i);
            l0CEventList[i] = i;
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[i]);
        }
    }

    __forceinline__ [aicore]
    void L1TileMmad(L1TileMmadParams const &params)
    {
        uint32_t mPartLoop = CeilDiv(params.mRound, (uint32_t)L0TileShape::M);
        uint32_t nPartLoop = CeilDiv(params.nRound, (uint32_t)L0TileShape::N);
        uint32_t kPartLoop = CeilDiv(params.kActual, (uint32_t)L0TileShape::K);
        auto &l1ATensor = l1ATensorList[params.l1ListId];
        auto &l1BTensor = l1BTensorList[params.l1ListId];
        auto &l0CTensor = l0CTensorList[l0CListId];
        LayoutCInL0 layoutCInL0 = LayoutCInL0::MakeLayoutInL0C(MakeCoord(params.mRound, params.nRound));

        if constexpr (!ENABLE_UNIT_FLAG) {
            if (params.isKLoopFirst) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            }
        }

        for (uint32_t mPartIdx = 0; mPartIdx < mPartLoop; ++mPartIdx) {
            uint32_t mPartActual = (mPartIdx < mPartLoop - 1) ?
                L0TileShape::M : (params.mRound - mPartIdx * L0TileShape::M);
            for (uint32_t kPartIdx = 0; kPartIdx < kPartLoop; ++kPartIdx) {
                uint32_t kPartActual = (kPartIdx < kPartLoop - 1) ?
                    L0TileShape::K : (params.kActual - kPartIdx * L0TileShape::K);
                auto &l0ATile = l0ATensorList[l0AListId];
                auto layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mPartActual, kPartActual);
                auto l1AOffset = MakeCoord(mPartIdx, kPartIdx) * L0TileShape::ToCoordMK();
                auto l1ATile = l1ATensor[L1A_LAYOUT.GetOffset(l1AOffset)];
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                if ((mPartIdx == 0) && (kPartIdx == 0)) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1AEventList[params.l1ListId]);
                }
                copyL1ToL0A(l0ATile, l1ATile, layoutAInL0, L1A_LAYOUT);
                if ((mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1)) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1AEventList[params.l1ListId]);
                }

                for (uint32_t nPartIdx = 0; nPartIdx < nPartLoop; ++nPartIdx) {
                    uint32_t nPartActual = (nPartIdx < nPartLoop - 1) ?
                        L0TileShape::N : (params.nRound - nPartIdx * L0TileShape::N);
                    auto &l0BTile = l0BTensorList[l0BListId];
                    auto layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kPartActual, nPartActual);
                    auto l1BOffset = MakeCoord(kPartIdx, nPartIdx) * L0TileShape::ToCoordKN();
                    auto l1BTile = l1BTensor[L1B_LAYOUT.GetOffset(l1BOffset)];

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    if ((kPartIdx == 0) && (nPartIdx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1BEventList[params.l1ListId]);
                    }
                    copyL1ToL0B(l0BTile, l1BTile, layoutBInL0, L1B_LAYOUT);
                    if ((kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1BEventList[params.l1ListId]);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    auto l0COffset = MakeCoord(mPartIdx, nPartIdx) * L0TileShape::ToCoordMN();
                    auto l0CTile = l0CTensor[layoutCInL0.GetOffset(l0COffset)];
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    
                    bool initC = (params.isKLoopFirst && (kPartIdx == 0));
                    uint8_t unitFlag = 0b00;
                    if constexpr (ENABLE_UNIT_FLAG) {
                        if (params.isKLoopLast &&
                            (mPartIdx == mPartLoop - 1) && (kPartIdx == kPartLoop - 1) && (nPartIdx == nPartLoop - 1)) {
                            unitFlag = 0b11;
                        } else {
                            unitFlag = 0b10;
                        }
                    }
                    tileMmad(l0CTile, l0ATile, l0BTile, mPartActual, nPartActual, kPartActual, initC, unitFlag);

                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0BEventList[l0BListId]);
                    l0BListId = (l0BListId + 1 < L0B_STAGES) ? (l0BListId + 1) : 0;
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0AEventList[l0AListId]);
                l0AListId = (l0AListId + 1 < L0A_STAGES) ? (l0AListId + 1) : 0;
            }
        }

        if (params.isKLoopLast) {
            auto layoutCInGm = params.layoutCInGm;

            if constexpr (!ENABLE_UNIT_FLAG) {
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CEventList[l0CListId]);
                copyL0CToGm(params.gmBlockC, l0CTensor, layoutCInGm, layoutCInL0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CEventList[l0CListId]);
            } else {
                copyL0CToGm(params.gmBlockC, l0CTensor, layoutCInGm, layoutCInL0, 0b11);
            }
            l0CListId = (l0CListId + 1 < L0C_STAGES) ? (l0CListId + 1) : 0;

            if (params.callback) {
                params.callback();
            }
        }
    }

    AscendC::LocalTensor<ElementA> l1ATensorList[L1_STAGES];
    AscendC::LocalTensor<ElementB> l1BTensorList[L1_STAGES];
    int32_t l1AEventList[L1_STAGES];
    int32_t l1BEventList[L1_STAGES];
    uint32_t l1ListId{0};

    AscendC::LocalTensor<ElementA> l0ATensorList[L0A_STAGES];
    int32_t l0AEventList[L0A_STAGES];
    uint32_t l0AListId{0};

    AscendC::LocalTensor<ElementB> l0BTensorList[L0B_STAGES];
    int32_t l0BEventList[L0B_STAGES];
    uint32_t l0BListId{0};

    AscendC::LocalTensor<ElementAccumulator> l0CTensorList[L0C_STAGES_];
    int32_t l0CEventList[L0C_STAGES_];
    uint32_t l0CListId{0};

    L1TileMmadParams l1TileMmadParamsList[PRELOAD_STAGES];
    uint32_t l1TileMmadParamsId{0};
    uint32_t preloadCount{0};

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;
};

}  // namespace Catlass::Gemm::Block

namespace Catlass::Gemm::Kernel {

template <
    class BlockMmad_,
    class BlockEpilogue_,
    class BlockScheduler_,
    class ElementGroupList_
>
class GroupedMatmulSliceM {
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
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using ElementGroupList = ElementGroupList_;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        uint32_t problemCount;
        __gm__ ElementGroupList *ptrGroupList;
        __gm__ ElementA *ptrA;
        LayoutA layoutA;
        __gm__ ElementB *ptrB;
        LayoutB layoutB;
        __gm__ ElementC *ptrC;
        LayoutC layoutC;

        // Methods
        __forceinline__ [aicore]
        Params() {}

        __forceinline__ [aicore]
        Params(
            GemmCoord const &problemShape_, uint32_t problemCount_, GM_ADDR ptrGroupList_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_
        ) : problemShape(problemShape_),
            problemCount(problemCount_), ptrGroupList(reinterpret_cast<__gm__ ElementGroupList *>(ptrGroupList_)),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrC(reinterpret_cast<__gm__ ElementC *>(ptrC_)), layoutC(layoutC_)
        {
        }
    };
    // Methods
    __forceinline__ [aicore]
    GroupedMatmulSliceM() {}
    // Methods
    __forceinline__ [aicore]
    ~GroupedMatmulSliceM(){}

    template <int32_t CORE_TYPE = g_coreType>
    __forceinline__ [aicore]
    void operator()(Params const &params);

    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler blockScheduler;
        Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);

        // Represent the full gm
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(params.ptrA);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(params.ptrC);
        AscendC::GlobalTensor<ElementGroupList> groupList;
        groupList.SetGlobalBuffer(params.ptrGroupList);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;
        for (uint32_t groupIdx = 0; groupIdx < params.problemCount; ++groupIdx) {
            uint32_t currentM = (groupIdx == 0) ? groupList.GetValue(groupIdx) :
                (groupList.GetValue(groupIdx) - groupList.GetValue(groupIdx - 1));
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};

            LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB = params.layoutB;
            LayoutC layoutC = params.layoutC.GetTileLayout(inGroupProblemShape.GetCoordMN());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            AscendC::GlobalTensor<ElementB> gmB;
            gmB.SetGlobalBuffer(params.ptrB + gmGroupOffsetB);
            if (CeilDiv(currentM, L1TileShape::M) == 1) {
                gmB.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx;
            if (coreIdx < startCoreIdx) {
                startLoopIdx = coreIdx + coreNum - startCoreIdx;
            } else {
                startLoopIdx = coreIdx - startCoreIdx;
            }
            // Loop through the matmul of each groupIdx
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);

                // Compute block-scoped matrix multiply-add
                blockMmad(
                    gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                    gmB[gmOffsetB], layoutB,
                    gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                    actualBlockShape
                );
            }

            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();

            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
    }

    template <>
    __forceinline__ [aicore]
    void operator()<AscendC::AIV>(Params const &params)
    {
    }
};

} // namespace Catlass::Gemm::Kernel

using namespace Catlass;
using namespace matmul;

template <
    class LayoutA,
    class LayoutB,
    class LayoutC
>
__forceinline__ [aicore]
void GroupedMatmulSliceM(
    GemmCoord problemShape,
    uint32_t problemCount, GM_ADDR gmGroupList,
    GM_ADDR gmA, LayoutA layoutA,
    GM_ADDR gmB, LayoutB layoutB,
    GM_ADDR gmC, LayoutC layoutC
)
{
    if (problemShape.k() > problemShape.n()) {
        constexpr uint32_t preloadStages = 1;
        constexpr uint32_t l1Stages = 2;
        constexpr uint32_t l0AStages = 2;
        constexpr uint32_t l0BStages = 4;
        constexpr uint32_t l0CStages = 1;
        constexpr bool enableUnitFlag = true;
        constexpr bool enableShuffleK = true;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
            preloadStages,
            l1Stages, l0AStages, l0BStages, l0CStages,
            enableUnitFlag, enableShuffleK
        >;
        using L1TileShape = GemmShape<256, 128, 256>;
        using L0TileShape = GemmShape<256, 128, 64>;

        using AType = Gemm::GemmType<half, LayoutA>;
        using BType = Gemm::GemmType<half, LayoutB>;
        using CType = Gemm::GemmType<half, LayoutC>;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockEpilogue = void;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;

        typename MatmulKernel::Params params{
            problemShape, problemCount, gmGroupList, gmA, layoutA, gmB, layoutB, gmC, layoutC
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);

    } else {
        constexpr uint32_t preloadStages = 1;
        constexpr uint32_t l1Stages = 2;
        constexpr uint32_t l0AStages = 4;
        constexpr uint32_t l0BStages = 2;
        constexpr uint32_t l0CStages = 1;
        constexpr bool enableUnitFlag = true;
        constexpr bool enableShuffleK = true;

        using ArchTag = Arch::AtlasA2;
        using DispatchPolicy = Gemm::MmadAtlasA2PreloadAsync<
            preloadStages,
            l1Stages, l0AStages, l0BStages, l0CStages,
            enableUnitFlag, enableShuffleK
        >;
        using L1TileShape = GemmShape<128, 256, 256>;
        using L0TileShape = GemmShape<128, 256, 64>;

        using AType = Gemm::GemmType<half, LayoutA>;
        using BType = Gemm::GemmType<half, LayoutB>;
        using CType = Gemm::GemmType<half, LayoutC>;

        using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
        using BlockEpilogue = void;
        using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

        // kernel level
        using MatmulKernel = Gemm::Kernel::GroupedMatmulSliceM<BlockMmad, BlockEpilogue, BlockScheduler, int64_t>;

        typename MatmulKernel::Params params{
            problemShape, problemCount, gmGroupList, gmA, layoutA, gmB, layoutB, gmC, layoutC
        };

        // call a kernel
        MatmulKernel matmul;
        matmul(params);
    }
}

extern "C" __global__ __aicore__ void grouped_matmul_slice_m(GM_ADDR a, GM_ADDR b, GM_ADDR groupList, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    uint32_t m = tiling_data.m;
    uint32_t k = tiling_data.k;
    uint32_t n = tiling_data.n;
    uint32_t groupCount = tiling_data.groupCount;
    GemmCoord problemShape{m, n, k};

    using LayoutA = layout::RowMajor;
    using LayoutB = layout::RowMajor;
    using LayoutC = layout::RowMajor;
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m, n};

    GroupedMatmulSliceM<LayoutA, LayoutB, LayoutC>(
        problemShape, groupCount, groupList,
        a, layoutA,
        b, layoutB,
        c, layoutC);
}