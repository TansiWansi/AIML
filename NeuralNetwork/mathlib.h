#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "arena.h"


#ifndef MATHLIB_H
#define MATHLIB_H

#define TILE_SIZE 64

#define BETA_V 0.999
#define BETA_M 0.9
#define EPSILON 1e-8

typedef uint32_t u32;
typedef float f32;

#define MAX(m, n)((m > n)? m : n)
#define MIN(m, n)((m < n)? m : n)

typedef enum{
	
	MAT_DEFAULT		= 0,
	MAT_TRANSPOSED	= 1 << 0,
	MAT_VIEW 		= 1 << 1
}MatrixFlags;

typedef struct Matrix{
	
	u32 rows;
	u32 cols;
	f32 * restrict data;
	u32 flags;
}Matrix;

Matrix *createMatrix(const u32 rows, const u32 cols);

void freeMatrix(Matrix *m);

Matrix *createArenaMatrix(Arena * restrict arena, const u32 rows, const u32 cols);

Matrix *matScale(const f32 scaleFactor, const Matrix * restrict A, Matrix * restrict result);

Matrix *matDot(const Matrix * restrict A, const Matrix * restrict B, Matrix * restrict result);

Matrix *matTranspose(const Matrix * restrict A, Matrix * restrict A_T);

Matrix *matAdd(const Matrix * restrict A, const Matrix * restrict B, Matrix * restrict result);

Matrix *matColSum(const Matrix * restrict A, Matrix * restrict result);

Matrix *reluInplace(Matrix * restrict A, const f32 leak);

Matrix *reluDerivInplace(Matrix * restrict grad, const Matrix * restrict inputs, const f32 leak);

Matrix *matAdamUpdate(Matrix * restrict W, const Matrix * restrict grad, Matrix * restrict M, Matrix * restrict V, const f32 lr, const f32 betaM, const f32 betaW, const f32 epsilon, const f32 l2lambda, u32 timeStep);


// mseLossGrad

#endif
