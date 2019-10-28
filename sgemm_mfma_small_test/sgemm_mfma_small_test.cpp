
#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
// hip header file
#include "hip/hip_runtime.h"

#define NUM_AB (32 * 16)
#define NUM_C (32 * 32)

/* MFMA definitions */
#define THREADS_PER_WAVE 64
typedef float mfma_float16 __attribute__((ext_vector_type(16)));
typedef __fp16 mfma_half4 __attribute__((ext_vector_type(4)));
extern "C" __device__ mfma_float16 __llvm_amdgcn_mfma_f32_32x32x8f16(mfma_half4, mfma_half4, mfma_float16, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x8f16");
extern "C" __device__ mfma_float16 __llvm_amdgcn_mfma_f32_32x32x2f32(float, float, mfma_float16, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2f32");


  /******************************************/
  /* Function Prefix                        */
  /******************************************/



/* tile parameters */
#define NUM_THREADS  64
#define SG0I 8
#define SG1J 8
#define TT0I 8
#define TT1J 8
#define MT0I (SG0I*TT0I)
#define MT1J (SG1J*TT1J)
#define VECTOR_WIDTH 1
#define GLOBAL_LOAD_VECTOR_WIDTH_A 1
#define GLOBAL_LOAD_VECTOR_WIDTH_B 1
#define GLOBAL_WRITE_VECTOR_WIDTH 1

/* DepthU parameters*/
#define CPSV (NUM_THREADS / MT0I * VECTOR_WIDTH)
#define LOCAL_SPLITU 1
#define UNROLL 8
#define LOCAL_DEPTHU (LOCAL_SPLITU*UNROLL)

/* other */
#define PAD 0
#define WORK_GROUP_MAPPING 1

/* num loads parallel and perpendicular to coalesced */
#define NLCA 1
#define NLCB 1
#define NLPA 8
#define NLPB 8

/* load sizes parallel and perpendicular to coalesced */
#define LSCA (MT0I/NLCA)
#define LSPA (LOCAL_DEPTHU/NLPA)
#define LSCB (LOCAL_DEPTHU/NLCB)
#define LSPB (MT1J/NLPB)
#define LVCA (LSCA/GLOBAL_LOAD_VECTOR_WIDTH_A)
#define LVCB (LSCB/GLOBAL_LOAD_VECTOR_WIDTH_B)
#define LVPA (LSPA/GLOBAL_LOAD_VECTOR_WIDTH_A)
#define LVPB (LSPB/GLOBAL_LOAD_VECTOR_WIDTH_B)
#define LDS_OFFSET_B 512
#define LDS_NUM_ELEMENTS 2048
#define LDS_OFFSET_BLK 1024

/* global memory indices */
#define GLOBAL_D(IDX0I, IDX1J, IDXK) (( (IDX0I)*strideD0I + (IDX1J)*strideD1J + (IDXK)*strideDK ))
#define GLOBAL_C(IDX0I, IDX1J, IDXK) (( (IDX0I)*strideC0I + (IDX1J)*strideC1J + (IDXK)*strideCK ))
#define GLOBAL_OFFSET_A(IDX0I, IDXL, IDXK) (( (IDX0I)*strideA0I + (IDXL)*strideAL + (IDXK)*strideAK ))
#define GLOBAL_OFFSET_B(IDXL, IDX1J, IDXK) (( (IDXL)*strideBL + (IDX1J)*strideB1J + (IDXK)*strideBK ))

/* data types */
#define DATA_TYPE float
#define DEST_DATA_TYPE float
#define COMPUTE_DATA_TYPE float
#define MAGIC_DIV(dividend, magicNumber, magicShift) ((uint64_t)(dividend) * magicNumber >> magicShift)

/* MAC's */
#define MAC(A,B,DST) DST += A*B
#define TYPE_MAC(MULA,MULB,DST) DST = MAC(MULA,MULB,DST);
#define TYPE_MAC_WRITE(DST,SRC,ALPHA,REG,BETA) DST = 0 != (BETA) ? (ALPHA)*(REG) + (BETA)*(SRC) : (ALPHA)*(REG);

/* 8x8 micro-tile */
#define MAC_8x8\
  TYPE_MAC(rA[0],rB[0],rC[0+0*TT0I]); \
  TYPE_MAC(rA[1],rB[0],rC[1+0*TT0I]); \
  TYPE_MAC(rA[2],rB[0],rC[2+0*TT0I]); \
  TYPE_MAC(rA[3],rB[0],rC[3+0*TT0I]); \
  TYPE_MAC(rA[4],rB[0],rC[4+0*TT0I]); \
  TYPE_MAC(rA[5],rB[0],rC[5+0*TT0I]); \
  TYPE_MAC(rA[6],rB[0],rC[6+0*TT0I]); \
  TYPE_MAC(rA[7],rB[0],rC[7+0*TT0I]); \
  TYPE_MAC(rA[0],rB[1],rC[0+1*TT0I]); \
  TYPE_MAC(rA[1],rB[1],rC[1+1*TT0I]); \
  TYPE_MAC(rA[2],rB[1],rC[2+1*TT0I]); \
  TYPE_MAC(rA[3],rB[1],rC[3+1*TT0I]); \
  TYPE_MAC(rA[4],rB[1],rC[4+1*TT0I]); \
  TYPE_MAC(rA[5],rB[1],rC[5+1*TT0I]); \
  TYPE_MAC(rA[6],rB[1],rC[6+1*TT0I]); \
  TYPE_MAC(rA[7],rB[1],rC[7+1*TT0I]); \
  TYPE_MAC(rA[0],rB[2],rC[0+2*TT0I]); \
  TYPE_MAC(rA[1],rB[2],rC[1+2*TT0I]); \
  TYPE_MAC(rA[2],rB[2],rC[2+2*TT0I]); \
  TYPE_MAC(rA[3],rB[2],rC[3+2*TT0I]); \
  TYPE_MAC(rA[4],rB[2],rC[4+2*TT0I]); \
  TYPE_MAC(rA[5],rB[2],rC[5+2*TT0I]); \
  TYPE_MAC(rA[6],rB[2],rC[6+2*TT0I]); \
  TYPE_MAC(rA[7],rB[2],rC[7+2*TT0I]); \
  TYPE_MAC(rA[0],rB[3],rC[0+3*TT0I]); \
  TYPE_MAC(rA[1],rB[3],rC[1+3*TT0I]); \
  TYPE_MAC(rA[2],rB[3],rC[2+3*TT0I]); \
  TYPE_MAC(rA[3],rB[3],rC[3+3*TT0I]); \
  TYPE_MAC(rA[4],rB[3],rC[4+3*TT0I]); \
  TYPE_MAC(rA[5],rB[3],rC[5+3*TT0I]); \
  TYPE_MAC(rA[6],rB[3],rC[6+3*TT0I]); \
  TYPE_MAC(rA[7],rB[3],rC[7+3*TT0I]); \
  TYPE_MAC(rA[0],rB[4],rC[0+4*TT0I]); \
  TYPE_MAC(rA[1],rB[4],rC[1+4*TT0I]); \
  TYPE_MAC(rA[2],rB[4],rC[2+4*TT0I]); \
  TYPE_MAC(rA[3],rB[4],rC[3+4*TT0I]); \
  TYPE_MAC(rA[4],rB[4],rC[4+4*TT0I]); \
  TYPE_MAC(rA[5],rB[4],rC[5+4*TT0I]); \
  TYPE_MAC(rA[6],rB[4],rC[6+4*TT0I]); \
  TYPE_MAC(rA[7],rB[4],rC[7+4*TT0I]); \
  TYPE_MAC(rA[0],rB[5],rC[0+5*TT0I]); \
  TYPE_MAC(rA[1],rB[5],rC[1+5*TT0I]); \
  TYPE_MAC(rA[2],rB[5],rC[2+5*TT0I]); \
  TYPE_MAC(rA[3],rB[5],rC[3+5*TT0I]); \
  TYPE_MAC(rA[4],rB[5],rC[4+5*TT0I]); \
  TYPE_MAC(rA[5],rB[5],rC[5+5*TT0I]); \
  TYPE_MAC(rA[6],rB[5],rC[6+5*TT0I]); \
  TYPE_MAC(rA[7],rB[5],rC[7+5*TT0I]); \
  TYPE_MAC(rA[0],rB[6],rC[0+6*TT0I]); \
  TYPE_MAC(rA[1],rB[6],rC[1+6*TT0I]); \
  TYPE_MAC(rA[2],rB[6],rC[2+6*TT0I]); \
  TYPE_MAC(rA[3],rB[6],rC[3+6*TT0I]); \
  TYPE_MAC(rA[4],rB[6],rC[4+6*TT0I]); \
  TYPE_MAC(rA[5],rB[6],rC[5+6*TT0I]); \
  TYPE_MAC(rA[6],rB[6],rC[6+6*TT0I]); \
  TYPE_MAC(rA[7],rB[6],rC[7+6*TT0I]); \
  TYPE_MAC(rA[0],rB[7],rC[0+7*TT0I]); \
  TYPE_MAC(rA[1],rB[7],rC[1+7*TT0I]); \
  TYPE_MAC(rA[2],rB[7],rC[2+7*TT0I]); \
  TYPE_MAC(rA[3],rB[7],rC[3+7*TT0I]); \
  TYPE_MAC(rA[4],rB[7],rC[4+7*TT0I]); \
  TYPE_MAC(rA[5],rB[7],rC[5+7*TT0I]); \
  TYPE_MAC(rA[6],rB[7],rC[6+7*TT0I]); \
  TYPE_MAC(rA[7],rB[7],rC[7+7*TT0I]); \

/* hard-coded initial strides */
#define strideD0I 1
#define strideC0I 1
#define strideA0I 1
#define strideBL 1

/* MFMA definitions */
#define MFMA_IN_THREAD_ELEMENTS 1
#define MFMA_OUT_THREAD_ELEMENTS 4
#define MFMA_L 2
#define MFMA_M_N_DIM_THREADS 32
#define WV0I 1
#define WV1J 1
#define WT0I 2
#define WT1J 2


  /******************************************/
  /* Begin Kernel                           */
  /******************************************/

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
extern "C"
__global__ void Cijk_Ailk_Bljk_SB_MT64x64x8_SE_K1(
  float *D,
  float const * __restrict__ C,
  float const * __restrict__ A,
  float const * __restrict__ B,
  float const alpha,
  float const beta,
  unsigned int const strideD1J,
  unsigned int const strideDK,
  unsigned int const strideC1J,
  unsigned int const strideCK,
  unsigned int const strideAL,
  unsigned int const strideAK,
  unsigned int const strideB1J,
  unsigned int const strideBK,
  unsigned int const size0I,
  unsigned int const size1J,
  unsigned int const sizeK,
  unsigned int const sizeL,
  unsigned int staggerUIterParm,
  unsigned int problemNumGroupTiles0,
  unsigned int problemNumGroupTiles1,
  unsigned int magicNumberProblemNumGroupTiles0 )
#pragma clang diagnostic pop

 {


  /******************************************/
  /* Allocate Resources                     */
  /******************************************/

  unsigned int serial = hc_get_workitem_id(0);
  unsigned int sgId = serial / (WV0I*WV1J*THREADS_PER_WAVE);
  unsigned int wave_serial = serial % THREADS_PER_WAVE;
  unsigned int wvId = serial / (THREADS_PER_WAVE*LOCAL_SPLITU);
#define SCALAR_ZERO (float)(0)
#define SCALAR_OOB_DATA SCALAR_ZERO
  /* registers for MAC's */
  mfma_float16 rC[WT0I*WT1J] = {0};
  float rA[WT0I];
  float rB[WT1J];

  /* registers for global->local */
  DATA_TYPE a_0_0_0_0;
  DATA_TYPE a_0_0_1_0;
  DATA_TYPE a_0_0_2_0;
  DATA_TYPE a_0_0_3_0;
  DATA_TYPE a_0_0_4_0;
  DATA_TYPE a_0_0_5_0;
  DATA_TYPE a_0_0_6_0;
  DATA_TYPE a_0_0_7_0;
  DATA_TYPE b_0_0_0_0;
  DATA_TYPE b_0_0_1_0;
  DATA_TYPE b_0_0_2_0;
  DATA_TYPE b_0_0_3_0;
  DATA_TYPE b_0_0_4_0;
  DATA_TYPE b_0_0_5_0;
  DATA_TYPE b_0_0_6_0;
  DATA_TYPE b_0_0_7_0;

  /* allocate local memory */
  __shared__ DATA_TYPE localMemory[LDS_NUM_ELEMENTS];


  /******************************************/
  /* Local Read Addresses                   */
  /******************************************/


  /* local read addresses: tile assignments a */

  unsigned int lr0I = wave_serial % MFMA_M_N_DIM_THREADS ;


  /* local read addresses: tile assignments b */

  unsigned int lr1J = wave_serial / MFMA_M_N_DIM_THREADS ;


  /* local read addresses: final offsets a */

  unsigned int localReadOffsetA = lr0I + (lr1J*MFMA_IN_THREAD_ELEMENTS + sgId*MFMA_L )*(MT0I+PAD) + wvId%WV0I*MFMA_M_N_DIM_THREADS;


  /* local read addresses: final offsets b */

  unsigned int localReadOffsetB = lr0I + (lr1J*MFMA_IN_THREAD_ELEMENTS + sgId*MFMA_L )*(MT1J+PAD) + wvId/WV0I*MFMA_M_N_DIM_THREADS + LDS_OFFSET_B;


  /* local read addresses: declare addresses a */

  DATA_TYPE *localReadA;


  /* local read addresses: declare addresses b */

  DATA_TYPE *localReadB;



  /******************************************/
  /* Begin setupNewTile                     */
  /******************************************/


  /* global read addresses: work-group */

  unsigned int wg0I = hc_get_group_id(0);
  unsigned int wg1J = hc_get_group_id(1);
  unsigned int nwg0I = hc_get_num_groups(0);
  unsigned int nwg1J = hc_get_num_groups(1);


  /* global read addresses: tile offset assignment a */

  unsigned int globalReadOffsetA0I = (serial%LVCA)*GLOBAL_LOAD_VECTOR_WIDTH_A + (wg0I)*MT0I;


  /* global read addresses: tile offset assignment b */

  unsigned int globalReadOffsetB1J = (serial/LVCB) + (wg1J)*MT1J;


  /* global read addresses: unroll assignment a */

  unsigned int globalReadOffsetAL = (serial/LVCA);


  /* global read addresses: unroll assignment b */

  unsigned int globalReadOffsetBL = (serial%LVCB)*GLOBAL_LOAD_VECTOR_WIDTH_B;


  /* global read addresses: other free assignments */

  unsigned int wgK = ( hc_get_group_id(2) ) % sizeK;


  /* global read addresses: tile offsets a */

  unsigned int flattenedOffsetA_0_0 = globalReadOffsetA0I + 0 + 0*LSCA;
  flattenedOffsetA_0_0 = (flattenedOffsetA_0_0 > (size0I-1)) ? (size0I-1):flattenedOffsetA_0_0;
  unsigned int globalReadOffsetA0I_0_0 = flattenedOffsetA_0_0;


  /* global read addresses: tile offsets b */

  unsigned int flattenedOffsetB_0_0 = globalReadOffsetB1J + 0 + 0*LSPB;
  flattenedOffsetB_0_0 = (flattenedOffsetB_0_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_0_0;
  unsigned int globalReadOffsetB1J_0_0 = flattenedOffsetB_0_0;
  unsigned int flattenedOffsetB_1_0 = globalReadOffsetB1J + 0 + 1*LSPB;
  flattenedOffsetB_1_0 = (flattenedOffsetB_1_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_1_0;
  unsigned int globalReadOffsetB1J_1_0 = flattenedOffsetB_1_0;
  unsigned int flattenedOffsetB_2_0 = globalReadOffsetB1J + 0 + 2*LSPB;
  flattenedOffsetB_2_0 = (flattenedOffsetB_2_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_2_0;
  unsigned int globalReadOffsetB1J_2_0 = flattenedOffsetB_2_0;
  unsigned int flattenedOffsetB_3_0 = globalReadOffsetB1J + 0 + 3*LSPB;
  flattenedOffsetB_3_0 = (flattenedOffsetB_3_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_3_0;
  unsigned int globalReadOffsetB1J_3_0 = flattenedOffsetB_3_0;
  unsigned int flattenedOffsetB_4_0 = globalReadOffsetB1J + 0 + 4*LSPB;
  flattenedOffsetB_4_0 = (flattenedOffsetB_4_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_4_0;
  unsigned int globalReadOffsetB1J_4_0 = flattenedOffsetB_4_0;
  unsigned int flattenedOffsetB_5_0 = globalReadOffsetB1J + 0 + 5*LSPB;
  flattenedOffsetB_5_0 = (flattenedOffsetB_5_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_5_0;
  unsigned int globalReadOffsetB1J_5_0 = flattenedOffsetB_5_0;
  unsigned int flattenedOffsetB_6_0 = globalReadOffsetB1J + 0 + 6*LSPB;
  flattenedOffsetB_6_0 = (flattenedOffsetB_6_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_6_0;
  unsigned int globalReadOffsetB1J_6_0 = flattenedOffsetB_6_0;
  unsigned int flattenedOffsetB_7_0 = globalReadOffsetB1J + 0 + 7*LSPB;
  flattenedOffsetB_7_0 = (flattenedOffsetB_7_0 > (size1J-1)) ? (size1J-1):flattenedOffsetB_7_0;
  unsigned int globalReadOffsetB1J_7_0 = flattenedOffsetB_7_0;


  /* global read addresses: unroll offsets a */

  unsigned int globalReadOffsetAL_0_0 = globalReadOffsetAL + 0 + 0*LSPA;
  unsigned int globalReadOffsetAL_1_0 = globalReadOffsetAL + 0 + 1*LSPA;
  unsigned int globalReadOffsetAL_2_0 = globalReadOffsetAL + 0 + 2*LSPA;
  unsigned int globalReadOffsetAL_3_0 = globalReadOffsetAL + 0 + 3*LSPA;
  unsigned int globalReadOffsetAL_4_0 = globalReadOffsetAL + 0 + 4*LSPA;
  unsigned int globalReadOffsetAL_5_0 = globalReadOffsetAL + 0 + 5*LSPA;
  unsigned int globalReadOffsetAL_6_0 = globalReadOffsetAL + 0 + 6*LSPA;
  unsigned int globalReadOffsetAL_7_0 = globalReadOffsetAL + 0 + 7*LSPA;


  /* global read addresses: unroll offsets b */

  unsigned int globalReadOffsetBL_0_0 = globalReadOffsetBL + 0 + 0*LSCB;


  /* global read addresses: shift a */

  globalReadOffsetA0I_0_0 = (  globalReadOffsetA0I_0_0 > size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+0) ? size0I-GLOBAL_LOAD_VECTOR_WIDTH_A+0 : globalReadOffsetA0I_0_0;


  /* global read addresses: shift b */

  globalReadOffsetB1J_0_0 = (  globalReadOffsetB1J_0_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_0_0;
  globalReadOffsetB1J_1_0 = (  globalReadOffsetB1J_1_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_1_0;
  globalReadOffsetB1J_2_0 = (  globalReadOffsetB1J_2_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_2_0;
  globalReadOffsetB1J_3_0 = (  globalReadOffsetB1J_3_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_3_0;
  globalReadOffsetB1J_4_0 = (  globalReadOffsetB1J_4_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_4_0;
  globalReadOffsetB1J_5_0 = (  globalReadOffsetB1J_5_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_5_0;
  globalReadOffsetB1J_6_0 = (  globalReadOffsetB1J_6_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_6_0;
  globalReadOffsetB1J_7_0 = (  globalReadOffsetB1J_7_0 > size1J-1) ? size1J-1 : globalReadOffsetB1J_7_0;


  /* global read addresses: final offsets a */

  uint64_t globalReadOffsetA_0_0_0_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_0_0, wgK );
  uint64_t globalReadOffsetA_0_0_1_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_1_0, wgK );
  uint64_t globalReadOffsetA_0_0_2_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_2_0, wgK );
  uint64_t globalReadOffsetA_0_0_3_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_3_0, wgK );
  uint64_t globalReadOffsetA_0_0_4_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_4_0, wgK );
  uint64_t globalReadOffsetA_0_0_5_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_5_0, wgK );
  uint64_t globalReadOffsetA_0_0_6_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_6_0, wgK );
  uint64_t globalReadOffsetA_0_0_7_0 = GLOBAL_OFFSET_A( globalReadOffsetA0I_0_0, globalReadOffsetAL_7_0, wgK );


  /* global read addresses: final offsets b */

  uint64_t globalReadOffsetB_0_0_0_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_0_0, wgK );
  uint64_t globalReadOffsetB_0_0_1_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_1_0, wgK );
  uint64_t globalReadOffsetB_0_0_2_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_2_0, wgK );
  uint64_t globalReadOffsetB_0_0_3_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_3_0, wgK );
  uint64_t globalReadOffsetB_0_0_4_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_4_0, wgK );
  uint64_t globalReadOffsetB_0_0_5_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_5_0, wgK );
  uint64_t globalReadOffsetB_0_0_6_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_6_0, wgK );
  uint64_t globalReadOffsetB_0_0_7_0 = GLOBAL_OFFSET_B( globalReadOffsetBL_0_0, globalReadOffsetB1J_7_0, wgK );


  /* global read addresses: addresses a */

  DATA_TYPE const *globalReadA_0_0_0_0 = A + globalReadOffsetA_0_0_0_0;
  DATA_TYPE const *globalReadA_0_0_1_0 = A + globalReadOffsetA_0_0_1_0;
  DATA_TYPE const *globalReadA_0_0_2_0 = A + globalReadOffsetA_0_0_2_0;
  DATA_TYPE const *globalReadA_0_0_3_0 = A + globalReadOffsetA_0_0_3_0;
  DATA_TYPE const *globalReadA_0_0_4_0 = A + globalReadOffsetA_0_0_4_0;
  DATA_TYPE const *globalReadA_0_0_5_0 = A + globalReadOffsetA_0_0_5_0;
  DATA_TYPE const *globalReadA_0_0_6_0 = A + globalReadOffsetA_0_0_6_0;
  DATA_TYPE const *globalReadA_0_0_7_0 = A + globalReadOffsetA_0_0_7_0;


  /* global read addresses: addresses b */

  DATA_TYPE const *globalReadB_0_0_0_0 = B + globalReadOffsetB_0_0_0_0;
  DATA_TYPE const *globalReadB_0_0_1_0 = B + globalReadOffsetB_0_0_1_0;
  DATA_TYPE const *globalReadB_0_0_2_0 = B + globalReadOffsetB_0_0_2_0;
  DATA_TYPE const *globalReadB_0_0_3_0 = B + globalReadOffsetB_0_0_3_0;
  DATA_TYPE const *globalReadB_0_0_4_0 = B + globalReadOffsetB_0_0_4_0;
  DATA_TYPE const *globalReadB_0_0_5_0 = B + globalReadOffsetB_0_0_5_0;
  DATA_TYPE const *globalReadB_0_0_6_0 = B + globalReadOffsetB_0_0_6_0;
  DATA_TYPE const *globalReadB_0_0_7_0 = B + globalReadOffsetB_0_0_7_0;


  /* global read addresses: increments a */

  int64_t globalReadIncAL = (int64_t)strideAL*LOCAL_DEPTHU;


  /* global read addresses: increments b */

  int64_t globalReadIncBL = (int64_t)strideBL*LOCAL_DEPTHU;


  /******************************************/
  /* Local Write Addresses                  */
  /******************************************/


  /* local write addresses: tile assignment A */
  unsigned int lwA0I = (serial%LVCA)*GLOBAL_LOAD_VECTOR_WIDTH_A;


  /* local write addresses: tile assignment B */
  unsigned int lwB1J = (serial/LVCB);


  /* local write addresses: unroll assignment A */
  unsigned int lwAL = (serial/LVCA);


  /* local write addresses: unroll assignment B */
  unsigned int lwBL = (serial%LVCB)*GLOBAL_LOAD_VECTOR_WIDTH_B;


  /* local write addresses: first offset a */

  unsigned int localWriteFirstOffsetA = lwA0I + lwAL*(MT0I+PAD);


  /* local write addresses: first offset b */

  unsigned int localWriteFirstOffsetB = lwB1J + lwBL*(MT1J+PAD) + LDS_OFFSET_B;


  /* local write addresses: final offsets A */
  unsigned int localWriteOffsetA_0_0_0_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 0*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_1_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 1*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_2_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 2*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_3_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 3*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_4_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 4*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_5_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 5*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_6_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 6*LSPA)*(MT0I+PAD);
  unsigned int localWriteOffsetA_0_0_7_0 = localWriteFirstOffsetA + (0 + 0*LSCA) + (0 + 7*LSPA)*(MT0I+PAD);


  /* local write addresses: final offsets B */
  unsigned int localWriteOffsetB_0_0_0_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 0*LSPB);
  unsigned int localWriteOffsetB_0_0_1_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 1*LSPB);
  unsigned int localWriteOffsetB_0_0_2_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 2*LSPB);
  unsigned int localWriteOffsetB_0_0_3_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 3*LSPB);
  unsigned int localWriteOffsetB_0_0_4_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 4*LSPB);
  unsigned int localWriteOffsetB_0_0_5_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 5*LSPB);
  unsigned int localWriteOffsetB_0_0_6_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 6*LSPB);
  unsigned int localWriteOffsetB_0_0_7_0 = localWriteFirstOffsetB + (0 + 0*LSCB)*(MT1J+PAD) + (0 + 7*LSPB);


  /* local write addresses: declare addresses A */
  DATA_TYPE *localWriteA_0_0_0_0;
  DATA_TYPE *localWriteA_0_0_1_0;
  DATA_TYPE *localWriteA_0_0_2_0;
  DATA_TYPE *localWriteA_0_0_3_0;
  DATA_TYPE *localWriteA_0_0_4_0;
  DATA_TYPE *localWriteA_0_0_5_0;
  DATA_TYPE *localWriteA_0_0_6_0;
  DATA_TYPE *localWriteA_0_0_7_0;


  /* local write addresses: declare addresses B */
  DATA_TYPE *localWriteB_0_0_0_0;
  DATA_TYPE *localWriteB_0_0_1_0;
  DATA_TYPE *localWriteB_0_0_2_0;
  DATA_TYPE *localWriteB_0_0_3_0;
  DATA_TYPE *localWriteB_0_0_4_0;
  DATA_TYPE *localWriteB_0_0_5_0;
  DATA_TYPE *localWriteB_0_0_6_0;
  DATA_TYPE *localWriteB_0_0_7_0;


  /* local write init pointers A */
  localWriteA_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_0_0);
  localWriteA_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_1_0);
  localWriteA_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_2_0);
  localWriteA_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_3_0);
  localWriteA_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_4_0);
  localWriteA_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_5_0);
  localWriteA_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_6_0);
  localWriteA_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_7_0);


  /* local write init pointers B */
  localWriteB_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_0);
  localWriteB_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_1_0);
  localWriteB_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_2_0);
  localWriteB_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_3_0);
  localWriteB_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_4_0);
  localWriteB_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_5_0);
  localWriteB_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_6_0);
  localWriteB_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_7_0);

  /* declare loop num iterations */

  unsigned int numIterL;


  /* Compute unroll loop num iter */
  numIterL = sizeL / LOCAL_DEPTHU;

  const unsigned origNumIter = numIterL;
  unsigned staggerUIter = (wg0I & staggerUIterParm);
  staggerUIter = (staggerUIter << 3); // shift so each stagger has 256-byte stride

  globalReadA_0_0_0_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_1_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_2_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_3_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_4_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_5_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_6_0 += (staggerUIter * globalReadIncAL); // apply stagger offset
  globalReadA_0_0_7_0 += (staggerUIter * globalReadIncAL); // apply stagger offset


  globalReadB_0_0_0_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_1_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_2_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_3_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_4_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_5_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_6_0 += (staggerUIter * globalReadIncBL); // apply stagger offset
  globalReadB_0_0_7_0 += (staggerUIter * globalReadIncBL); // apply stagger offset

  staggerUIter += 1; // add PrefetchGlobalRead

  /* local read addresses: init pointers a */

  localReadA = (DATA_TYPE *)(localMemory + localReadOffsetA);

  /* local read addresses: init pointers b */

  localReadB = (DATA_TYPE *)(localMemory + localReadOffsetB);


  /* prefetch: global -> local */

  if (sizeL >= LOCAL_DEPTHU) {


    /* global read A */
    a_0_0_0_0 = *(globalReadA_0_0_0_0 + 0);
    a_0_0_1_0 = *(globalReadA_0_0_1_0 + 0);
    a_0_0_2_0 = *(globalReadA_0_0_2_0 + 0);
    a_0_0_3_0 = *(globalReadA_0_0_3_0 + 0);
    a_0_0_4_0 = *(globalReadA_0_0_4_0 + 0);
    a_0_0_5_0 = *(globalReadA_0_0_5_0 + 0);
    a_0_0_6_0 = *(globalReadA_0_0_6_0 + 0);
    a_0_0_7_0 = *(globalReadA_0_0_7_0 + 0);


    /* global read B */
    b_0_0_0_0 = *(globalReadB_0_0_0_0 + 0);
    b_0_0_1_0 = *(globalReadB_0_0_1_0 + 0);
    b_0_0_2_0 = *(globalReadB_0_0_2_0 + 0);
    b_0_0_3_0 = *(globalReadB_0_0_3_0 + 0);
    b_0_0_4_0 = *(globalReadB_0_0_4_0 + 0);
    b_0_0_5_0 = *(globalReadB_0_0_5_0 + 0);
    b_0_0_6_0 = *(globalReadB_0_0_6_0 + 0);
    b_0_0_7_0 = *(globalReadB_0_0_7_0 + 0);


    /* global read inc A */
    globalReadA_0_0_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_0_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_0_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_1_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_1_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_1_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_2_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_2_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_2_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_3_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_3_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_3_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_4_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_4_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_4_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_5_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_5_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_5_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_6_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_6_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_6_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_7_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_7_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_7_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }


    /* global read inc B */
    globalReadB_0_0_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_0_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_0_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_1_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_1_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_1_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_2_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_2_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_2_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_3_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_3_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_3_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_4_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_4_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_4_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_5_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_5_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_5_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_6_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_6_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_6_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_7_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_7_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_7_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }


    /******************************************/
    /* End setupNewTile                       */
    /******************************************/








    /* local write a */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
    *(localWriteA_0_0_0_0 + 0) = a_0_0_0_0;
    *(localWriteA_0_0_1_0 + 0) = a_0_0_1_0;
    *(localWriteA_0_0_2_0 + 0) = a_0_0_2_0;
    *(localWriteA_0_0_3_0 + 0) = a_0_0_3_0;
    *(localWriteA_0_0_4_0 + 0) = a_0_0_4_0;
    *(localWriteA_0_0_5_0 + 0) = a_0_0_5_0;
    *(localWriteA_0_0_6_0 + 0) = a_0_0_6_0;
    *(localWriteA_0_0_7_0 + 0) = a_0_0_7_0;
#pragma clang diagnostic pop


    /* local write b */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
    *(localWriteB_0_0_0_0 + 0) = b_0_0_0_0;
    *(localWriteB_0_0_1_0 + 0) = b_0_0_1_0;
    *(localWriteB_0_0_2_0 + 0) = b_0_0_2_0;
    *(localWriteB_0_0_3_0 + 0) = b_0_0_3_0;
    *(localWriteB_0_0_4_0 + 0) = b_0_0_4_0;
    *(localWriteB_0_0_5_0 + 0) = b_0_0_5_0;
    *(localWriteB_0_0_6_0 + 0) = b_0_0_6_0;
    *(localWriteB_0_0_7_0 + 0) = b_0_0_7_0;
#pragma clang diagnostic pop


    /* local write swap a */

    localWriteOffsetA_0_0_0_0 = (localWriteOffsetA_0_0_0_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_1_0 = (localWriteOffsetA_0_0_1_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_2_0 = (localWriteOffsetA_0_0_2_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_3_0 = (localWriteOffsetA_0_0_3_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_4_0 = (localWriteOffsetA_0_0_4_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_5_0 = (localWriteOffsetA_0_0_5_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_6_0 = (localWriteOffsetA_0_0_6_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_7_0 = (localWriteOffsetA_0_0_7_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);


    /* local write swap b */

    localWriteOffsetB_0_0_0_0 = (localWriteOffsetB_0_0_0_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_1_0 = (localWriteOffsetB_0_0_1_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_2_0 = (localWriteOffsetB_0_0_2_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_3_0 = (localWriteOffsetB_0_0_3_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_4_0 = (localWriteOffsetB_0_0_4_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_5_0 = (localWriteOffsetB_0_0_5_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_6_0 = (localWriteOffsetB_0_0_6_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_7_0 = (localWriteOffsetB_0_0_7_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);


    /* local write init pointers A */
    localWriteA_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_0_0);
    localWriteA_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_1_0);
    localWriteA_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_2_0);
    localWriteA_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_3_0);
    localWriteA_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_4_0);
    localWriteA_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_5_0);
    localWriteA_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_6_0);
    localWriteA_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_7_0);


    /* local write init pointers B */
    localWriteB_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_0);
    localWriteB_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_1_0);
    localWriteB_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_2_0);
    localWriteB_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_3_0);
    localWriteB_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_4_0);
    localWriteB_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_5_0);
    localWriteB_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_6_0);
    localWriteB_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_7_0);

  } // end PrefetchGlobalRead
  else { // still need to initC even if skipped prefetch

  }


  /******************************************/
  /* Unrolled Loop(s) - Begin               */
  /******************************************/

  while (numIterL-- > 1) {


    /******************************************/
    /* Unroll Loop 1/1 - Begin                */
    /******************************************/


    __syncthreads(); //4sync for global read


    /* global read A */
    a_0_0_0_0 = *(globalReadA_0_0_0_0 + 0);
    a_0_0_1_0 = *(globalReadA_0_0_1_0 + 0);
    a_0_0_2_0 = *(globalReadA_0_0_2_0 + 0);
    a_0_0_3_0 = *(globalReadA_0_0_3_0 + 0);
    a_0_0_4_0 = *(globalReadA_0_0_4_0 + 0);
    a_0_0_5_0 = *(globalReadA_0_0_5_0 + 0);
    a_0_0_6_0 = *(globalReadA_0_0_6_0 + 0);
    a_0_0_7_0 = *(globalReadA_0_0_7_0 + 0);

    /* global read B */
    b_0_0_0_0 = *(globalReadB_0_0_0_0 + 0);
    b_0_0_1_0 = *(globalReadB_0_0_1_0 + 0);
    b_0_0_2_0 = *(globalReadB_0_0_2_0 + 0);
    b_0_0_3_0 = *(globalReadB_0_0_3_0 + 0);
    b_0_0_4_0 = *(globalReadB_0_0_4_0 + 0);
    b_0_0_5_0 = *(globalReadB_0_0_5_0 + 0);
    b_0_0_6_0 = *(globalReadB_0_0_6_0 + 0);
    b_0_0_7_0 = *(globalReadB_0_0_7_0 + 0);

    /* global read inc A */
    globalReadA_0_0_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_0_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_0_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_1_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_1_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_1_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_2_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_2_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_2_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_3_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_3_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_3_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_4_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_4_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_4_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_5_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_5_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_5_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_6_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_6_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_6_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }
    globalReadA_0_0_7_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadA_0_0_7_0) + globalReadIncAL);
    if ((numIterL) == staggerUIter) {
      globalReadA_0_0_7_0 -= (origNumIter * globalReadIncAL); // wrap staggered offset back to row start
    }

    /* global read inc B */
    globalReadB_0_0_0_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_0_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_0_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_1_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_1_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_1_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_2_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_2_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_2_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_3_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_3_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_3_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_4_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_4_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_4_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_5_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_5_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_5_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_6_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_6_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_6_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }
    globalReadB_0_0_7_0 = (DATA_TYPE const *)( ((DATA_TYPE const *)globalReadB_0_0_7_0) + globalReadIncBL);
    if ((numIterL) == staggerUIter) {
      globalReadB_0_0_7_0 -= (origNumIter * globalReadIncBL); // wrap staggered offset back to row start
    }





    /* iter 0 */


    /* local read a */
    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 

    /* local read b */
    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);
    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );



    /* iter 1 */


    /* local read a */
    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 

    /* local read b */
    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);
    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );



    /* iter 2 */


    /* local read a */
    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 

    /* local read b */
    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 

    /* local read increment a */
    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);

    /* local read increment b */
    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);
    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );





    /* iter 3 (last) */


    /* local read a */
    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 

    /* local read b */
    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 

/* local write A */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
    *(localWriteA_0_0_0_0 + 0) = a_0_0_0_0;
    *(localWriteA_0_0_1_0 + 0) = a_0_0_1_0;
    *(localWriteA_0_0_2_0 + 0) = a_0_0_2_0;
    *(localWriteA_0_0_3_0 + 0) = a_0_0_3_0;
    *(localWriteA_0_0_4_0 + 0) = a_0_0_4_0;
    *(localWriteA_0_0_5_0 + 0) = a_0_0_5_0;
    *(localWriteA_0_0_6_0 + 0) = a_0_0_6_0;
    *(localWriteA_0_0_7_0 + 0) = a_0_0_7_0;
#pragma clang diagnostic pop

/* local write B */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
    *(localWriteB_0_0_0_0 + 0) = b_0_0_0_0;
    *(localWriteB_0_0_1_0 + 0) = b_0_0_1_0;
    *(localWriteB_0_0_2_0 + 0) = b_0_0_2_0;
    *(localWriteB_0_0_3_0 + 0) = b_0_0_3_0;
    *(localWriteB_0_0_4_0 + 0) = b_0_0_4_0;
    *(localWriteB_0_0_5_0 + 0) = b_0_0_5_0;
    *(localWriteB_0_0_6_0 + 0) = b_0_0_6_0;
    *(localWriteB_0_0_7_0 + 0) = b_0_0_7_0;
#pragma clang diagnostic pop

    /* local write swap offsets a */
    localWriteOffsetA_0_0_0_0 = (localWriteOffsetA_0_0_0_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_1_0 = (localWriteOffsetA_0_0_1_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_2_0 = (localWriteOffsetA_0_0_2_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_3_0 = (localWriteOffsetA_0_0_3_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_4_0 = (localWriteOffsetA_0_0_4_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_5_0 = (localWriteOffsetA_0_0_5_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_6_0 = (localWriteOffsetA_0_0_6_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetA_0_0_7_0 = (localWriteOffsetA_0_0_7_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);

    /* local write swap offsets b */
    localWriteOffsetB_0_0_0_0 = (localWriteOffsetB_0_0_0_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_1_0 = (localWriteOffsetB_0_0_1_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_2_0 = (localWriteOffsetB_0_0_2_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_3_0 = (localWriteOffsetB_0_0_3_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_4_0 = (localWriteOffsetB_0_0_4_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_5_0 = (localWriteOffsetB_0_0_5_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_6_0 = (localWriteOffsetB_0_0_6_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);
    localWriteOffsetB_0_0_7_0 = (localWriteOffsetB_0_0_7_0 + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);

    /* local write init pointers A */
    localWriteA_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_0_0);
    localWriteA_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_1_0);
    localWriteA_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_2_0);
    localWriteA_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_3_0);
    localWriteA_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_4_0);
    localWriteA_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_5_0);
    localWriteA_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_6_0);
    localWriteA_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_7_0);

    /* local write init pointers B */
    localWriteB_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_0);
    localWriteB_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_1_0);
    localWriteB_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_2_0);
    localWriteB_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_3_0);
    localWriteB_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_4_0);
    localWriteB_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_5_0);
    localWriteB_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_6_0);
    localWriteB_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_7_0);

    /* local read swap offsets a */
    localReadOffsetA = (localReadOffsetA + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);

    /* local read swap offsets b */
    localReadOffsetB = (localReadOffsetB + LDS_OFFSET_BLK)%(LDS_OFFSET_BLK*2);

    /* local read init pointers a */
    localReadA = (DATA_TYPE *)(localMemory + localReadOffsetA);

    /* local read init pointers b */
    localReadB = (DATA_TYPE *)(localMemory + localReadOffsetB);
    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );



    /******************************************/
    /* Unrolled Loop - End                    */
    /******************************************/

  }


  if (sizeL >= LOCAL_DEPTHU) {


    __syncthreads(); //


    /* iter 0 */


    /* local read a */

    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 


    /* local read b */

    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 


    /* local read inc a */

    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);


    /* local read inc b */

    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);


    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );



    /* iter 1 */


    /* local read a */

    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 


    /* local read b */

    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 


    /* local read inc a */

    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);


    /* local read inc b */

    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);


    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );



    /* iter 2 */


    /* local read a */

    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 


    /* local read b */

    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 


    /* local read inc a */

    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);


    /* local read inc b */

    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);


    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );



    /* iter 3 */


    /* local read a */

    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 


    /* local read b */

    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 


    /* local read inc a */

    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);


    /* local read inc b */

    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);


    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );


  } // end unroll


  /******************************************/
  /* Tail Loop                              */
  /******************************************/


  /* local write reset offsets a */

  localWriteOffsetA_0_0_0_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_1_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_2_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_3_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_4_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_5_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_6_0 %= LDS_OFFSET_BLK;
  localWriteOffsetA_0_0_7_0 %= LDS_OFFSET_BLK;


  /* local write reset offsets b */

  localWriteOffsetB_0_0_0_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_1_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_2_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_3_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_4_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_5_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_6_0 %= LDS_OFFSET_BLK;
  localWriteOffsetB_0_0_7_0 %= LDS_OFFSET_BLK;


  /* Compute tail loop num iter */
  numIterL = (((sizeL % LOCAL_DEPTHU) + (LOCAL_SPLITU*MFMA_L) - 1) / (LOCAL_SPLITU*MFMA_L));


  /* remove stagger offsets for tail loop */

  globalReadA_0_0_0_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_1_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_2_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_3_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_4_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_5_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_6_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset
  globalReadA_0_0_7_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncAL); // remove stagger offset

  globalReadB_0_0_0_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_1_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_2_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_3_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_4_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_5_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_6_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset
  globalReadB_0_0_7_0 += ((origNumIter - (staggerUIter - 1)) * globalReadIncBL); // remove stagger offset


  /* global read a */


  /* global read A */
  a_0_0_0_0 = ( globalReadOffsetAL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_0_0 + 0);
  a_0_0_1_0 = ( globalReadOffsetAL_1_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_1_0 + 0);
  a_0_0_2_0 = ( globalReadOffsetAL_2_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_2_0 + 0);
  a_0_0_3_0 = ( globalReadOffsetAL_3_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_3_0 + 0);
  a_0_0_4_0 = ( globalReadOffsetAL_4_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_4_0 + 0);
  a_0_0_5_0 = ( globalReadOffsetAL_5_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_5_0 + 0);
  a_0_0_6_0 = ( globalReadOffsetAL_6_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_6_0 + 0);
  a_0_0_7_0 = ( globalReadOffsetAL_7_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadA_0_0_7_0 + 0);


  /* global read b */


  /* global read B */
  b_0_0_0_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_0_0 + 0);
  b_0_0_1_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_1_0 + 0);
  b_0_0_2_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_2_0 + 0);
  b_0_0_3_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_3_0 + 0);
  b_0_0_4_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_4_0 + 0);
  b_0_0_5_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_5_0 + 0);
  b_0_0_6_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_6_0 + 0);
  b_0_0_7_0 = ( globalReadOffsetBL_0_0 + 0 >= (sizeL % LOCAL_DEPTHU) ) ? SCALAR_OOB_DATA : *(globalReadB_0_0_7_0 + 0);


  __syncthreads(); //


  /* local write init pointers A */
  localWriteA_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_0_0);
  localWriteA_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_1_0);
  localWriteA_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_2_0);
  localWriteA_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_3_0);
  localWriteA_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_4_0);
  localWriteA_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_5_0);
  localWriteA_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_6_0);
  localWriteA_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetA_0_0_7_0);


  /* local write init pointers B */
  localWriteB_0_0_0_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_0_0);
  localWriteB_0_0_1_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_1_0);
  localWriteB_0_0_2_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_2_0);
  localWriteB_0_0_3_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_3_0);
  localWriteB_0_0_4_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_4_0);
  localWriteB_0_0_5_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_5_0);
  localWriteB_0_0_6_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_6_0);
  localWriteB_0_0_7_0 = (DATA_TYPE *)(localMemory + localWriteOffsetB_0_0_7_0);


  /* local write a */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
  *(localWriteA_0_0_0_0 + 0) = a_0_0_0_0;
  *(localWriteA_0_0_1_0 + 0) = a_0_0_1_0;
  *(localWriteA_0_0_2_0 + 0) = a_0_0_2_0;
  *(localWriteA_0_0_3_0 + 0) = a_0_0_3_0;
  *(localWriteA_0_0_4_0 + 0) = a_0_0_4_0;
  *(localWriteA_0_0_5_0 + 0) = a_0_0_5_0;
  *(localWriteA_0_0_6_0 + 0) = a_0_0_6_0;
  *(localWriteA_0_0_7_0 + 0) = a_0_0_7_0;
#pragma clang diagnostic pop


  /* local write b */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
  *(localWriteB_0_0_0_0 + 0) = b_0_0_0_0;
  *(localWriteB_0_0_1_0 + 0) = b_0_0_1_0;
  *(localWriteB_0_0_2_0 + 0) = b_0_0_2_0;
  *(localWriteB_0_0_3_0 + 0) = b_0_0_3_0;
  *(localWriteB_0_0_4_0 + 0) = b_0_0_4_0;
  *(localWriteB_0_0_5_0 + 0) = b_0_0_5_0;
  *(localWriteB_0_0_6_0 + 0) = b_0_0_6_0;
  *(localWriteB_0_0_7_0 + 0) = b_0_0_7_0;
#pragma clang diagnostic pop


  __syncthreads(); //


  /* local read reset offsets a */

  localReadOffsetA %= LDS_OFFSET_BLK;


  /* local read reset offsets b */

  localReadOffsetB %= LDS_OFFSET_BLK;


  /* local read init pointers a */

  localReadA = (DATA_TYPE *)(localMemory + localReadOffsetA);


  /* local read init pointers b */

  localReadB = (DATA_TYPE *)(localMemory + localReadOffsetB);


  /* tail loop: macs */

  while (numIterL-- > 0) {


    /* local read a */

    rA[0] = localReadA[0*MT0I + 0*WV0I*MFMA_M_N_DIM_THREADS]; 
    rA[1] = localReadA[0*MT0I + 1*WV0I*MFMA_M_N_DIM_THREADS]; 


    /* local read b */

    rB[0] = localReadB[0*MT1J + 0*WV1J*MFMA_M_N_DIM_THREADS]; 
    rB[1] = localReadB[0*MT1J + 1*WV1J*MFMA_M_N_DIM_THREADS]; 


    /* local read inc a */

    localReadA += LOCAL_SPLITU*MFMA_L*(MT0I+PAD);


    /* local read inc b */

    localReadB += LOCAL_SPLITU*MFMA_L*(MT1J+PAD);


    rC[0] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[0], rC[0],0,0,0 );
    rC[1] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[0], rC[1],0,0,0 );
    rC[2] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[0], rB[1], rC[2],0,0,0 );
    rC[3] = __llvm_amdgcn_mfma_f32_32x32x2f32(rA[1], rB[1], rC[3],0,0,0 );


  }




  /* not-LocalSplitU: global write indices */

  unsigned int flattenedGlobalC0 = (wg0I)*MT0I + (wvId % WV0I)*MFMA_M_N_DIM_THREADS + (wave_serial / MFMA_M_N_DIM_THREADS)*MFMA_OUT_THREAD_ELEMENTS;
  unsigned int flattenedGlobalC1 = (wg1J)*MT1J + (wvId / WV0I)*MFMA_M_N_DIM_THREADS + (wave_serial % MFMA_M_N_DIM_THREADS);
  unsigned int globalC0I = flattenedGlobalC0;
  unsigned int globalC1J = flattenedGlobalC1;
  unsigned int globalCK = (wgK);


  /* not-LocalSplitU: global write */


  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][0 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][1 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][2 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][3 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][0 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][1 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][2 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][3 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][0 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][1 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][2 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][3 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][0 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][1 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][2 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][3 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][0 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][1 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][2 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][3 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][0 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][1 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][2 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][3 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][0 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][1 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][2 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 0*WT0I][3 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][0 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][1 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][2 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 0*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 0*WT0I][3 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][0 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][1 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][2 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][3 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][0 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][1 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][2 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 0*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][3 + (0*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][0 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][1 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][2 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][3 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][0 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][1 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][2 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 1*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][3 + (1*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][0 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][1 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][2 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][3 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][0 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][1 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][2 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 2*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][3 + (2*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][0 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][1 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][2 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 0*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[0 + 1*WT0I][3 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 0 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][0 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 1 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][1 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 2 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][2 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }

  /* new vw0 offset - inc and extract tensor dims */
  globalC0I =   flattenedGlobalC0 + 3 + 3*2*MFMA_OUT_THREAD_ELEMENTS + 1*WV0I*MFMA_M_N_DIM_THREADS;
  /* new vw1 offset - inc and extract tensor dims */
  globalC1J =   flattenedGlobalC1 + 1*WV1J*MFMA_M_N_DIM_THREADS;
  if (globalC0I < size0I) {  if (globalC1J < size1J) {  TYPE_MAC_WRITE( D[ GLOBAL_D( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], C[ GLOBAL_C( (uint64_t) globalC0I, (uint64_t) globalC1J, (uint64_t) globalCK) ], alpha, rC[1 + 1*WT0I][3 + (3*MFMA_OUT_THREAD_ELEMENTS)], beta) } }


}
// ==================================== Kernel end ===========================================

int main() {
    DATA_TYPE* cpu_A;
    DATA_TYPE* cpu_B;
    DATA_TYPE* cpu_C;
    DATA_TYPE* cpu_final;

    DATA_TYPE* gpu_A;
    DATA_TYPE* gpu_B;
    DATA_TYPE* gpu_C;

    DATA_TYPE alpha;
    DATA_TYPE beta;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    int i;

    cpu_A = (DATA_TYPE*)malloc(NUM_AB * sizeof(DATA_TYPE));
    cpu_B = (DATA_TYPE*)malloc(NUM_AB * sizeof(DATA_TYPE));
    cpu_C = (DATA_TYPE*)malloc(NUM_C * sizeof(DATA_TYPE));
    cpu_final = (DATA_TYPE*)malloc(NUM_C * sizeof(DATA_TYPE));

    // initialize the input data
    for (i = 0; i < NUM_AB; i++) {
        cpu_A[i] = (DATA_TYPE) 1.0f;
        cpu_B[i] = (DATA_TYPE) -1.0f;
    }
    for (i = 0; i < NUM_C; i++) {
        cpu_C[i] = (DATA_TYPE) 0;
    }

//Debug
    std::cout << " A Matrix: ";
    for (i = 0; i < NUM_AB; i++)
      std::cout << cpu_A[i] << ",";
    std::cout << std::endl;

    std::cout << " B Matrix: ";
    for (i = 0; i < NUM_AB; i++)
      std::cout << cpu_B[i] << ",";
    std::cout << std::endl;

    alpha = (DATA_TYPE) 1.0f;
    beta = (DATA_TYPE) 0;

    // allocate the memory on the device side
    hipMalloc((void**)&gpu_A, NUM_AB * sizeof(DATA_TYPE));
    hipMalloc((void**)&gpu_B, NUM_AB * sizeof(DATA_TYPE));
    hipMalloc((void**)&gpu_C, NUM_C * sizeof(DATA_TYPE));

    // Memory transfer from host to device
    hipMemcpy(gpu_A, cpu_A, NUM_AB * sizeof(DATA_TYPE), hipMemcpyHostToDevice);
    hipMemcpy(gpu_B, cpu_B, NUM_AB * sizeof(DATA_TYPE), hipMemcpyHostToDevice);
    hipMemcpy(gpu_C, cpu_C, NUM_C * sizeof(DATA_TYPE), hipMemcpyHostToDevice);

    // Lauching kernel from host
    hipLaunchKernelGGL(
	HIP_KERNEL_NAME(Cijk_Ailk_Bljk_SB_MT64x64x8_SE_K1), 
	dim3(1,  1, 1),
	dim3(64, 1, 1),
        0, // groupMemBytes
        0,//stream
        gpu_C,
        gpu_C,
        gpu_B,
        gpu_A,
        alpha,
        beta,
        32,//stride begin
        1024,
        32,
        1024,
        32,
        512,
        16,
        512,//stride end
        32, //size i
        32, //size j
        1, //size k 
        16, //size l
        0,//sataggerUIter
        1,//problemNumGroupTiles0
        1,//problemNumGroupTiles1
        1//magicNumberProblemNumGroupTiles0
        );

    // Memory transfer from device to host
    hipMemcpy(cpu_final, gpu_C, NUM_C * sizeof(DATA_TYPE), hipMemcpyDeviceToHost);

    // verify the results

    std::cout << "Result = " << cpu_final[0] << std::endl;
    if(cpu_final[0] == -16.0f)
      std::cout << "PASS! " << std::endl;
    else
      std::cout << "FAILED! " << std::endl;
    // free the resources on device side
    hipFree(gpu_A);
    hipFree(gpu_B);
    hipFree(gpu_C);

    // free the resources on host side
    free(cpu_A);
    free(cpu_B);
    free(cpu_C);
    free(cpu_final);

    return 0;
}
