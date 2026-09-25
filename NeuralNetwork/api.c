#include <stdint.h>
#include <stdio.h>
#include "engine.h"

Arena *persistantArena;
Arena *scratchArena;
NeuralNetwork *nn;

Matrix *_wrapNumpyArray(f32 * restrict data, const u32 rows, const u32 cols){
	
	Matrix *m = (Matrix *)arenaAlloc(persistantArena, sizeof(Matrix));

	*m = (Matrix){
		.rows  = rows,
		.cols  = cols,
		.data  = data,
		.flags = MAT_VIEW
	};

	return m;
}

void initEngine(u32 * restrict layerSizes, const u32 numLayers, const f32 l2lambda){

	openblas_set_num_threads(1);

	persistantArena = arenaInit(100 * 1024 * 1024);

	scratchArena 	= arenaInit(50 * 1024 * 1024);

	Architecture *arch = newArchitecture(persistantArena, layerSizes, numLayers, l2lambda);

	nn = newNeuralNetwork(persistantArena, arch);

}

void train(f32 * restrict xRaw, f32 * restrict yRaw, const u32 rows, const u32 xCols, const u32 yCols, const u32 epochs, const f32 lr, const u32 batchSize){
	
	if(!nn){
		printf("Neural Network not initialized!\n");
		return;
	}

	Matrix *xTrain = _wrapNumpyArray(xRaw, rows, xCols);
	Matrix *yTrain = _wrapNumpyArray(yRaw, rows, yCols);
	
	trainNeuralNetwork(nn, xTrain, yTrain, epochs, lr, batchSize, scratchArena);
}

void cleanupEngine(void){
	
	arenaDestroy(persistantArena);
	arenaDestroy(scratchArena);
	nn = NULL;
}

void saveWeights(const char *filepath){
	
	if(!nn){ printf("Engine not initialized!\n"); return; }

	FILE *f = fopen(filepath, "wb");
	if(!f){
		printf("File could not be opened!\n");
		return;
	}

	for(u32 i = 0; i < nn->numLayers; i++){
		
		Layer l = nn->layers[i];

		if(l.type == DENSE_LAYER){
			DenseLayer *dl = l.layer.dense;
			u32 wSize = dl->w->rows * dl->w->cols;
			u32 bSize = dl->b->rows * dl->b->cols;

			fwrite(dl->w->data, sizeof(f32), wSize, f);
			fwrite(dl->b->data, sizeof(f32), bSize, f);
		}
	}

	fclose(f);
	printf("Model weights saved to %s\n", filepath);
}


void loadWeights(const char *filepath){
	
	if(!nn){ printf("Engine not initialized!\n"); return; }

	FILE *f = fopen(filepath, "rb");
	if(!f){
		printf("File could not be opened!\n");
		return;
	}

	for(u32 i = 0; i < nn->numLayers; i++){
		
		Layer l = nn->layers[i];

		if(l.type == DENSE_LAYER){
			DenseLayer *dl = l.layer.dense;
			u32 wSize = dl->w->rows * dl->w->cols;
			u32 bSize = dl->b->rows * dl->b->cols;

			// 1. Read and verify weights
   		 	size_t wRead = fread(dl->w->data, sizeof(f32), wSize, f);
   		 	if (wRead != wSize) {
    		    fprintf(stderr, "FATAL: Failed to read weights! Expected %u elements, but got %lu.\n", wSize, wRead);
    		    exit(1);
    		}
	
   			 // 2. Read and verify biases
    		size_t bRead = fread(dl->b->data, sizeof(f32), bSize, f);
    		if (bRead != bSize) {
    	  	  fprintf(stderr, "FATAL: Failed to read biases! Expected %u elements, but got %lu.\n", bSize, bRead);
    		    exit(1);
    		}
		}
	}

	fclose(f);
	printf("Model weights loaded from %s\n", filepath);
}

void predict(f32 *x_raw, f32 *y_pred_raw, u32 rows, u32 x_cols, u32 y_cols) {
    if (nn == NULL) {
        printf("FATAL: C Engine not initialized!\n");
        return;
    }

    // 1. Wrap the Python inputs and the empty Python output buffer on the C Stack
    Matrix X_test = { .rows = rows, .cols = x_cols, .data = x_raw, .flags = MAT_VIEW };
    Matrix Y_pred = { .rows = rows, .cols = y_cols, .data = y_pred_raw, .flags = MAT_VIEW };

    // 2. Run the forward pass (Allocates intermediate math in the Scratch Arena)
    Matrix *predictions = predictNeuralNetwork(nn, &X_test, scratchArena);

    // 3. Copy the final results directly into Python's memory space
    u32 total_elements = rows * y_cols;
    for (u32 i = 0; i < total_elements; i++) {
        Y_pred.data[i] = predictions->data[i];
    }

    // 4. Instantly wipe the scratch memory so we are ready for the next predict call!
    arenaReset(scratchArena);
}
