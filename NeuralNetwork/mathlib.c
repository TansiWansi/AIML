#include "mathlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h> 

Matrix *createMatrix(const u32 rows, const u32 cols) {
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if(m == NULL) {
        printf("FATAL : Matrix allocation failed\n");
        exit(1);
    }
    *m = (Matrix){
        .rows = rows,
        .cols = cols,
        .flags = 0, 
        .data = (f32 *)calloc(rows * cols, sizeof(f32))
    };
    if(m->data == NULL){
        printf("FATAL : Matrix data initialization failed\n");
        exit(1);
    }
    return m;
}

void freeMatrix(Matrix *m) {
    // DO NOT free if the memory belongs to Python (MAT_VIEW)
    if(m) {
        if(m->data && !(m->flags & MAT_VIEW)) { 
            free(m->data); 
        }
        free(m);
    }
}

Matrix *createArenaMatrix(Arena * restrict arena, const u32 rows, const u32 cols) {

    Matrix *m = (Matrix *)arenaAlloc(arena, sizeof(Matrix));
    *m = (Matrix){
        .rows = rows,
        .cols = cols,
        .flags = 0,
        .data = (f32 *)arenaCalloc(arena, rows * cols * sizeof(f32))
    };
    return m;
}

Matrix *matScale(const f32 scaleFactor, const Matrix * restrict A, Matrix * restrict result) {
	
    u32 total = A->rows * A->cols;
    for(u32 i = 0; i < total; i++){ result->data[i] = A->data[i] * scaleFactor; }
    return result;
}


Matrix *matDot(const Matrix * restrict A, const Matrix * restrict B, Matrix * restrict result) {
    

    CBLAS_TRANSPOSE transA = (A->flags & MAT_TRANSPOSED) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transB = (B->flags & MAT_TRANSPOSED) ? CblasTrans : CblasNoTrans;

    u32 M = (transA == CblasTrans) ? A->cols : A->rows;
    u32 K = (transA == CblasTrans) ? A->rows : A->cols;
    u32 N = (transB == CblasTrans) ? B->rows : B->cols;

    u32 lda = A->cols;
    u32 ldb = B->cols;
    u32 ldc = result->cols;

    if (A->data == NULL || B->data == NULL || result->data == NULL) {
        printf("[FATAL] matDot received a NULL data pointer!\n");
        exit(1);
    }


    cblas_sgemm(
        CblasRowMajor, transA, transB, 
        M, N, K, 
        1.0f, A->data, lda, 
        B->data, ldb, 
        0.0f, result->data, ldc
    );


    return result;
}


Matrix *matAdd(const Matrix * restrict A, const Matrix * restrict B, Matrix * restrict result) {

	
    if (B->rows == 1 && B->cols == A->cols) {
        for(u32 i = 0; i < A->rows; i++) {
            for(u32 j = 0; j < A->cols; j++) {
                result->data[i * A->cols + j] = A->data[i * A->cols + j] + B->data[j];
            }
        }
    } else if (A->rows == B->rows && A->cols == B->cols) {
        u32 total = A->rows * A->cols;
        for(u32 i = 0; i < total; i++) {
            result->data[i] = A->data[i] + B->data[i];
        }
    } else {
        printf("\n[FATAL] matAdd Mismatch: A(%u, %u) + B(%u, %u)\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }
    return result;
}

Matrix *matColSum(const Matrix * restrict A, Matrix * restrict result) {


    for(u32 j = 0; j < A->cols; j++) {
        f32 sum = 0.0f;
        for(u32 i = 0; i < A->rows; i++) {
            sum += A->data[i * A->cols + j];
        }
        result->data[j] = sum;
    }
    return result;
}

Matrix *reluInplace(Matrix * restrict A, const f32 leak) {
    u32 total = A->rows * A->cols;
    for(u32 i = 0; i < total; i++){
        A->data[i] = (A->data[i] < 0.0f) ? A->data[i] * leak : A->data[i]; 
    }
    return A;
}

Matrix *reluDerivInplace(Matrix * restrict grad, const Matrix * restrict inputs, const f32 leak) {
    u32 total = grad->rows * grad->cols;
    for(u32 i = 0; i < total; i++){
        grad->data[i] *= (inputs->data[i] < 0.0f) ? leak : 1.0f;
    }
    return grad; 
}

Matrix *matAdamUpdate(Matrix * restrict W, const Matrix * restrict grad, Matrix * restrict M, Matrix * restrict V, const f32 lr, const f32 betaM, const f32 betaV, const f32 epsilon, const f32 l2lambda, u32 timeStep) {


    u32 total = W->rows * W->cols;
    f32 mCorr = 1.0f - powf(betaM, (f32)timeStep);
    f32 vCorr = 1.0f - powf(betaV, (f32)timeStep);

    for(u32 i = 0; i < total; i++){
        f32 wVal = W->data[i];
        f32 gVal = grad->data[i] + (l2lambda * wVal);

        f32 mVal = (betaM * M->data[i]) + ((1.0f - betaM) * gVal);
        f32 vVal = (betaV * V->data[i]) + ((1.0f - betaV) * (gVal * gVal));

        M->data[i] = mVal;
        V->data[i] = vVal;

        f32 mHat = mVal / mCorr;
        f32 vHat = vVal / vCorr;

        W->data[i] = wVal - (lr * mHat / (sqrtf(vHat) + epsilon));
    }
    return W;
}
