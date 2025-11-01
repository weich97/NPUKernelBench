#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2; 

struct TilingParam{
    uint32_t total_length;
    uint32_t start_length;
    uint32_t end_length;
    uint32_t weight_length;
    uint32_t ALIGN_NUM;
    uint32_t tiling_size;
    uint32_t block_size;
    uint32_t core_size;
    uint32_t core_remain;
    uint32_t mode;
    uint32_t shape[20];
    uint32_t reduce1[20];
    uint32_t reduce2[20];
    uint32_t reduce3[20];
    uint32_t dim;
};

template<typename TYPE_S, typename TYPE_E, typename TYPE_W, typename TYPE_Y> class Lerp {
public:
    __aicore__ inline Lerp() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight,GM_ADDR y, TilingParam& paramList) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = paramList.core_size + (GetBlockNum() == GetBlockIdx() + 1 ? paramList.core_remain : 0);
        this->tileLength = paramList.block_size;
        uint32_t ALIGN_NUM = paramList.ALIGN_NUM
        ASSERT(ALIGN_NUM != 0 && "ALIGN_NUM can not be zero!");
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->mode = paramList.mode;

        auto startPointer = paramList.core_size * GetBlockIdx();
        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        startGm.SetGlobalBuffer((__gm__ TYPE_S*)start + startPointer, bufferlength);
        endGm.SetGlobalBuffer((__gm__ TYPE_E*)end + startPointer, bufferlength);

        if constexpr (!std::is_same_v<TYPE_S, float>) {
            pipe.InitBuffer(calbuf1, this->tileLength * sizeof(float));
            pipe.InitBuffer(calbuf2, this->tileLength * sizeof(float));
            pipe.InitBuffer(calbuf3, this->tileLength * sizeof(float));
            pipe.InitBuffer(calbuf4, this->tileLength * sizeof(float));
        }else if(mode==0){
            pipe.InitBuffer(calbuf3, this->tileLength * sizeof(float));
        }

        if(mode==1){
            weightGm.SetGlobalBuffer((__gm__ TYPE_W*)weight + startPointer, bufferlength);
            pipe.InitBuffer(inQueueWEIGHT, BUFFER_NUM, this->tileLength * sizeof(TYPE_W));
        }else{
            weightGm.SetGlobalBuffer((__gm__ TYPE_W*)weight + startPointer, 1);
            this->weight = weightGm.GetValue(0);
            this->weightTmp = calbuf3.Get<float>();
            Duplicate(this->weightTmp,this->weight,this->tileLength);
        }

        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);

        // pipe alloc memory to queue, the unit is Bytes
        pipe.InitBuffer(inQueueSTART, BUFFER_NUM, this->tileLength * sizeof(TYPE_S));
        pipe.InitBuffer(inQueueEND, BUFFER_NUM, this->tileLength * sizeof(TYPE_E));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        if(this->mode==0){
            if(std::is_same_v<TYPE_S, float>){
                for (int32_t i = 0; i < loopCount-1; i++) {
                    CopyIn_mode0(i, this->tileLength);
                    Computefp32_mode0(i, this->tileLength);
                    CopyOut(i, this->tileLength);
                }
                auto length = this->blockLength - this->tileLength * (loopCount - 1);
                CopyIn_mode0(loopCount - 1, length);
                Computefp32_mode0(loopCount - 1, length);
                CopyOut(loopCount - 1, length);
            }else{
                for (int32_t i = 0; i < loopCount-1; i++) {
                    CopyIn_mode0(i, this->tileLength);
                    Computefp16_mode0(i, this->tileLength);
                    CopyOut(i, this->tileLength);
                }
                auto length = this->blockLength - this->tileLength * (loopCount - 1);
                CopyIn_mode0(loopCount - 1, length);
                Computefp16_mode0(loopCount - 1, length);
                CopyOut(loopCount - 1, length);
            }
        }else if(this->mode==1){
                if(std::is_same_v<TYPE_S, float>){
                    for (int32_t i = 0; i < loopCount-1; i++) {
                        CopyIn_mode1(i, this->tileLength);
                        Computefp32_mode1(i, this->tileLength);
                        CopyOut(i, this->tileLength);
                    }
                    auto length = this->blockLength - this->tileLength * (loopCount - 1);
                    CopyIn_mode1(loopCount - 1, length);
                    Computefp32_mode1(loopCount - 1, length);
                    CopyOut(loopCount - 1, length);
                }else{
                    for (int32_t i = 0; i < loopCount-1; i++) {
                        CopyIn_mode1(i, this->tileLength);
                        Computefp16_mode1(i, this->tileLength);
                        CopyOut(i, this->tileLength);
                    }
                    auto length = this->blockLength - this->tileLength * (loopCount - 1);
                    CopyIn_mode1(loopCount - 1, length);
                    Computefp16_mode1(loopCount - 1, length);
                    CopyOut(loopCount - 1, length);
                }
        }
    }
private:
    __aicore__ inline void CopyIn_mode0(int32_t progress, uint32_t length) {//weight len 1
        LocalTensor<DTYPE_START> startLocal = inQueueSTART.AllocTensor<TYPE_S>();
        LocalTensor<DTYPE_END> endLocal = inQueueEND.AllocTensor<TYPE_E>();

        DataCopy(startLocal, startGm[progress * this->tileLength], length);
        DataCopy(endLocal, endGm[progress * this->tileLength], length);

        inQueueSTART.EnQue(startLocal);
        inQueueEND.EnQue(endLocal);
    }
    __aicore__ inline void CopyIn_mode1(int32_t progress, uint32_t length) {//weight len > 1
        LocalTensor<DTYPE_START> startLocal = inQueueSTART.AllocTensor<TYPE_S>();
        LocalTensor<DTYPE_END> endLocal = inQueueEND.AllocTensor<TYPE_E>();
        LocalTensor<DTYPE_WEIGHT> weightLocal = inQueueWEIGHT.AllocTensor<TYPE_W>();

        DataCopy(startLocal, startGm[progress * this->tileLength], length);
        DataCopy(endLocal, endGm[progress * this->tileLength], length);
        DataCopy(weightLocal, weightGm[progress * this->tileLength], length);

        inQueueSTART.EnQue(startLocal);
        inQueueEND.EnQue(endLocal);
        inQueueWEIGHT.EnQue(weightLocal);
    }
    __aicore__ inline void Computefp32_mode0(int32_t progress, uint32_t length) {
        LocalTensor<float> startLocal = inQueueSTART.DeQue<float>();
        LocalTensor<float> endLocal = inQueueEND.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        Sub(yLocal,endLocal,startLocal,length);
        Mul(yLocal,this->weightTmp,yLocal,length);
        Add(yLocal,yLocal,startLocal,length);

        outQueueY.EnQue<float>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
    }
    __aicore__ inline void Computefp16_mode0(int32_t progress, uint32_t length) {
        LocalTensor<half> startLocal = inQueueSTART.DeQue<half>();
        LocalTensor<half> endLocal = inQueueEND.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        fstart = calbuf1.Get<float>();
        fend = calbuf2.Get<float>();
        auto fy = calbuf4.Get<float>();

        Cast(fstart, startLocal, RoundMode::CAST_NONE, length);
        Cast(fend, endLocal, RoundMode::CAST_NONE, length);

        Sub(fy, fend, fstart, length);
        Mul(fy, this->weightTmp, fy, length);
        Add(fy, fy, fstart, length);
        Cast(yLocal, fy, RoundMode::CAST_NONE, length);

        outQueueY.EnQue<half>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
    }
    __aicore__ inline void Computefp32_mode1(int32_t progress, uint32_t length) {
        LocalTensor<float> startLocal = inQueueSTART.DeQue<float>();
        LocalTensor<float> endLocal = inQueueEND.DeQue<float>();
        LocalTensor<float> weightLocal = inQueueWEIGHT.DeQue<float>();
        LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();

        Sub(yLocal,endLocal,startLocal,length);
        Mul(yLocal,weightLocal,yLocal,length);
        Add(yLocal,yLocal,startLocal,length);

        outQueueY.EnQue<float>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
        inQueueWEIGHT.FreeTensor(weightLocal);
    }
    __aicore__ inline void Computefp16_mode1(int32_t progress, uint32_t length) {
        LocalTensor<half> startLocal = inQueueSTART.DeQue<half>();
        LocalTensor<half> endLocal = inQueueEND.DeQue<half>();
        LocalTensor<half> weightLocal = inQueueWEIGHT.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        
        fstart = calbuf1.Get<float>();
        fend = calbuf2.Get<float>();
        fweight = calbuf3.Get<float>();
        auto fy = calbuf4.Get<float>();
        CastFunc(startLocal,endLocal,weightLocal,length);

        Sub(fy, fend, fstart, length);
        Mul(fy, fweight, fy, length);
        Add(fy, fy, fstart, length);
        Cast(yLocal, fy, RoundMode::CAST_NONE, length);

        outQueueY.EnQue<half>(yLocal);
        inQueueSTART.FreeTensor(startLocal);
        inQueueEND.FreeTensor(endLocal);
        inQueueWEIGHT.FreeTensor(weightLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CastFunc(LocalTensor<half> &startLocal, LocalTensor<half> &endLocal, LocalTensor<half> &weightLocal,
                                uint32_t length){
        Cast(fstart, startLocal, RoundMode::CAST_NONE, length);
        Cast(fend, endLocal, RoundMode::CAST_NONE, length);
        Cast(fweight, weightLocal, RoundMode::CAST_NONE, length);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSTART,inQueueEND,inQueueWEIGHT;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<QuePosition::VECCALC> calbuf1;
    TBuf<QuePosition::VECCALC> calbuf2;
    TBuf<QuePosition::VECCALC> calbuf3;
    TBuf<QuePosition::VECCALC> calbuf4;

    GlobalTensor<TYPE_S> startGm;
    GlobalTensor<TYPE_E> endGm;
    GlobalTensor<TYPE_W> weightGm;
    GlobalTensor<TYPE_Y> yGm;

    LocalTensor<float> fstart;
    LocalTensor<float> fend;
    LocalTensor<float> fweight;

    LocalTensor<float> weightTmp;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t mode;
    float weight;
};

template<typename TYPE_S, typename TYPE_E, typename TYPE_W, typename TYPE_Y> class Lerp_Broadcast {
public:
    __aicore__ inline Lerp_Broadcast() {}
    __aicore__ inline void Init(GM_ADDR start, GM_ADDR end, GM_ADDR weight,GM_ADDR y, TilingParam& paramList) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = paramList.core_size + (GetBlockNum() == GetBlockIdx() + 1 ? paramList.core_remain : 0);
        this->tileLength = paramList.block_size;
        this->ALIGN_NUM = paramList.ALIGN_NUM;
        ASSERT(ALIGN_NUM != 0 && "ALIGN_NUM can not be zero!");
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);
        this->startPointer = paramList.core_size * GetBlockIdx();

        this->reduce1 = paramList.reduce1;
        this->reduce2 = paramList.reduce2;
        this->reduce3 = paramList.reduce3;
        this->shape = paramList.shape;
        this->dim = paramList.dim;

        this->mode = paramList.mode;
        this->totalLength = paramList.total_length;
        this->startLength = paramList.start_length;
        this->endLength = paramList.end_length;
        this->weightLength = paramList.weight_length;

        auto bufferlength = this->blockLength;

        // get start index for current core, core parallel
        startGmB.SetGlobalBuffer((__gm__ TYPE_S*)start + startPointer, startLength);
        endGmB.SetGlobalBuffer((__gm__ TYPE_E*)end + startPointer, weightLength);
        weightGmB.SetGlobalBuffer((__gm__ TYPE_W*)weight + startPointer, endLength);
        yGmB.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);

        pipe.InitBuffer(inQueueSTARTB, BUFFER_NUM, this->tileLength * sizeof(TYPE_S));
        pipe.InitBuffer(inQueueENDB, BUFFER_NUM, this->tileLength * sizeof(TYPE_E));
        pipe.InitBuffer(inQueueWEIGHTB, BUFFER_NUM, this->tileLength * sizeof(TYPE_W));
        pipe.InitBuffer(outQueueYB, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        if constexpr (!std::is_same_v<TYPE_S, float>) {
            pipe.InitBuffer(calbufB1,this->tileLength * sizeof(float));
            pipe.InitBuffer(calbufB2,this->tileLength * sizeof(float));
            pipe.InitBuffer(calbufB3,this->tileLength * sizeof(float));
            pipe.InitBuffer(calbufB4,this->tileLength * sizeof(float));
        }
    }
    __aicore__ inline void Process() {
        int32_t count = this->totalLength / this->shape[this->dim - 1];
        uint32_t totalLength = this->shape[this->dim - 1];
        this->tileNum = totalLength / this->tileLength + (totalLength % this->tileLength > 0);
        uint32_t d[21] = {0};
        uint32_t dn1[21] = {0};
        uint32_t dn2[21] = {0};
        uint32_t dn3[21] = {0};

        auto dim = this->dim - 1;
        d[dim] = dn1[dim] = dn2[dim] = dn3[dim] = 1;
        InitializeDnArrays(d, dn1, this->reduce1, dim, this->shape);
        InitializeDnArrays(d, dn2, this->reduce2, dim, this->shape);
        InitializeDnArrays(d, dn3, this->reduce3, dim, this->shape);
        if(!std::is_same_v<TYPE_S, float>){
            for(int j=0;j<count;j++){
                uint32_t start1 = 0, start2 = 0, start3 = 0;
                CalculateStart(j, start1, dn1, reduce1, d);
                CalculateStart(j, start2, dn2, reduce2, d);
                CalculateStart(j, start3, dn3, reduce3, d);
                int32_t loopCount = this->tileNum;
                for (int32_t i = 0; i < loopCount-1; i++) {
                    CopyIn(start1 * totalLength, start2 * totalLength,start3 * totalLength, i, this->tileLength);
                    Computefp16(i, this->tileLength);
                    CopyOut(j * totalLength, i, this->tileLength);
                }
                uint32_t length = totalLength - this->tileLength * (loopCount - 1);
                auto length_align = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                CopyIn(start1 * totalLength, start2 * totalLength, start3 * totalLength, loopCount - 1, length_align);
                Computefp16(loopCount - 1, length);
                // Adds 31 to 'length' to round it up to the nearest multiple of 32 for alignment.
                CopyOut(j * totalLength, loopCount - 1, (length + 31) / 32 * 32);
            }   
        }else{
            for(int j=0;j<count;j++){
                uint32_t start1 = 0, start2 = 0, start3 = 0;
                CalculateStart(j, start1, dn1, reduce1, d);
                CalculateStart(j, start2, dn2, reduce2, d);
                CalculateStart(j, start3, dn3, reduce3, d);
                int32_t loopCount = this->tileNum;
                for (int32_t i = 0; i < loopCount-1; i++) {
                    CopyIn(start1 * totalLength, start2 * totalLength,start3 * totalLength, i, this->tileLength);
                    Computefp32(i, this->tileLength);
                    CopyOut(j * totalLength, i, this->tileLength);
                }
                uint32_t length = totalLength - this->tileLength * (loopCount - 1);
                auto length_align = (length + this->ALIGN_NUM - 1) / this->ALIGN_NUM * this->ALIGN_NUM;
                CopyIn(start1 * totalLength, start2 * totalLength, start3 * totalLength, loopCount - 1, length_align);
                Computefp32(loopCount - 1, length);
                // Adds 31 to 'length' to round it up to the nearest multiple of 32 for alignment.
                CopyOut(j * totalLength, loopCount - 1, (length + 31) / 32 * 32);
            } 
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t start1, uint32_t start2, uint32_t start3, int32_t progress, uint32_t length) {//both 1，n->m,n
        LocalTensor<TYPE_S> startLocal = inQueueSTARTB.AllocTensor<TYPE_S>();
        LocalTensor<TYPE_E> endLocal = inQueueENDB.AllocTensor<TYPE_E>();
        LocalTensor<TYPE_W> weightLocal = inQueueWEIGHTB.AllocTensor<TYPE_W>();

        DataCopy(startLocal, startGmB[start1 + progress * this->tileLength], length);
        DataCopy(endLocal, endGmB[start2 + progress * this->tileLength], length);
        DataCopy(weightLocal, weightGmB[start3 + progress * this->tileLength], length);
    
        inQueueSTARTB.EnQue(startLocal);
        inQueueENDB.EnQue(endLocal);
        inQueueWEIGHTB.EnQue(weightLocal);
    }
    __aicore__ inline void Computefp32(uint32_t progress, uint32_t length) {
        LocalTensor<TYPE_S> startLocalB = inQueueSTARTB.DeQue<TYPE_S>();
        LocalTensor<TYPE_E> endLocalB = inQueueENDB.DeQue<TYPE_E>();
        LocalTensor<TYPE_W> weightLocalB = inQueueWEIGHTB.DeQue<TYPE_W>();
        LocalTensor<TYPE_Y> yLocalB = outQueueYB.AllocTensor<TYPE_Y>();

        Sub(yLocalB,endLocalB,startLocalB,length);
        Mul(yLocalB,weightLocalB,yLocalB,length);
        Add(yLocalB,yLocalB,startLocalB,length);

        outQueueYB.EnQue<TYPE_Y>(yLocalB);
        inQueueSTARTB.FreeTensor(startLocalB);
        inQueueENDB.FreeTensor(endLocalB);
        inQueueWEIGHTB.FreeTensor(weightLocalB);
    }
    __aicore__ inline void Computefp16(uint32_t progress, uint32_t length) {
        LocalTensor<half> startLocalB = inQueueSTARTB.DeQue<half>();
        LocalTensor<half> endLocalB = inQueueENDB.DeQue<half>();
        LocalTensor<half> weightLocalB = inQueueWEIGHTB.DeQue<half>();
        LocalTensor<half> yLocalB = outQueueYB.AllocTensor<half>();

        fstartB = calbufB1.Get<float>();
        fendB = calbufB2.Get<float>();
        fweightB = calbufB3.Get<float>();
        auto fyB = calbufB4.Get<float>();
        CastFunc(startLocalB,endLocalB,weightLocalB,length);

        Sub(fyB, fendB, fstartB, length);
        Mul(fyB, fweightB, fyB, length);
        Add(fyB, fyB, fstartB, length);
        
        Cast(yLocalB, fyB, RoundMode::CAST_NONE, length);

        outQueueYB.EnQue<half>(yLocalB);
        inQueueSTARTB.FreeTensor(startLocalB);
        inQueueENDB.FreeTensor(endLocalB);
        inQueueWEIGHTB.FreeTensor(weightLocalB);
    }
    __aicore__ inline void CopyOut(uint32_t start,uint32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> yLocal = outQueueYB.DeQue<TYPE_Y>();
        DataCopy(yGmB[start + progress * this->tileLength], yLocal, length);
        outQueueYB.FreeTensor(yLocal);
    }
    __aicore__ inline void CalculateStart(int j, uint32_t &start, uint32_t* dn, uint32_t* reduce, uint32_t* d) {
        for (int k = dim - 1; k >= 0; k--) {
            uint32_t index = (j / d[k + 1] % shape[k]);

            if (reduce[k] == 0) {
                start += dn[k + 1] * index;
            }
        }
    }
    __aicore__ inline void InitializeDnArrays(uint32_t* d, uint32_t* dn, uint32_t* reduce, uint32_t dim, uint32_t* shape) {
        d[dim] = dn[dim] = 1;
        for (int k = dim - 1; k >= 0; k--) {
            d[k] = d[k + 1] * shape[k];
            if (reduce[k] == 0) {
                dn[k] = dn[k + 1] * shape[k];
            } else {
                dn[k] = dn[k + 1];
            }
        }
    }
    __aicore__ inline void CastFunc(LocalTensor<half> &startLocalB, LocalTensor<half> &endLocalB, LocalTensor<half> &weightLocalB,
                                uint32_t length){
        Cast(fstartB, startLocalB, RoundMode::CAST_NONE, length);
        Cast(fendB, endLocalB, RoundMode::CAST_NONE, length);
        Cast(fweightB, weightLocalB, RoundMode::CAST_NONE, length);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSTARTB;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueENDB;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueWEIGHTB;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueYB;
    TBuf<QuePosition::VECCALC> calbufB1;
    TBuf<QuePosition::VECCALC> calbufB2;
    TBuf<QuePosition::VECCALC> calbufB3;
    TBuf<QuePosition::VECCALC> calbufB4;

    GlobalTensor<TYPE_S> startGmB;
    GlobalTensor<TYPE_E> endGmB;
    GlobalTensor<TYPE_W> weightGmB;
    GlobalTensor<TYPE_Y> yGmB;

    LocalTensor<float> fstartB;
    LocalTensor<float> fendB;
    LocalTensor<float> fweightB;

    uint32_t blockLength;
    uint32_t startLength;
    uint32_t startPointer;
    uint32_t weightLength;
    uint32_t endLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t mode;
    uint32_t position;
    uint32_t totalLength;
    float weight;
    uint32_t* reduce1;
    uint32_t* reduce2;
    uint32_t* reduce3;
    uint32_t* shape;
    uint32_t dim;
    uint32_t ALIGN_NUM;
};

extern "C" __global__ __aicore__ void lerp(GM_ADDR start,
                                           GM_ADDR end,
                                           GM_ADDR weight,
                                           GM_ADDR y,
                                           GM_ADDR workspace,
                                           GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TilingParam paramList = {
        .total_length = tiling_data.total_length,
        .start_length = tiling_data.start_length,
        .end_length = tiling_data.end_length,
        .weight_length = tiling_data.weight_length,
        .ALIGN_NUM = tiling_data.ALIGN_NUM,
        .tiling_size = tiling_data.tiling_size,
        .block_size = tiling_data.block_size,
        .core_size = tiling_data.core_size,
        .core_remain = tiling_data.core_remain,
        .mode = tiling_data.mode,
        .dim = tiling_data.dim
    };
    for (int i = 0; i < 20; ++i) {
        paramList.shape[i] = tiling_data.shape[i];
        paramList.reduce1[i] = tiling_data.reduce1[i];
        paramList.reduce2[i] = tiling_data.reduce2[i];
        paramList.reduce3[i] = tiling_data.reduce3[i];
    }

    if (TILING_KEY_IS(0)){
        Lerp<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
        op.Init(start, end, weight, y, paramList);
        op.Process();
    }
    else if(TILING_KEY_IS(1)){
        Lerp_Broadcast<DTYPE_START, DTYPE_END, DTYPE_WEIGHT, DTYPE_Y> op;
        op.Init(start, end, weight, y, paramList);
        op.Process();
    }
}