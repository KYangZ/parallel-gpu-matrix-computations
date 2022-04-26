#include <stdio.h>
#include <stdlib.h>
#include "io.h"

int* read_file(FILE* fp, size_t* size){
    int index, value, capacity=10;
    index=0;
    int *array = (int*)malloc(sizeof(int)*capacity);
    while(fscanf(fp, " %d , ", &value)!=EOF){
        if(index>=capacity)
            array = (int*)realloc(array, sizeof(int)*(capacity*=2));
        array[index++] = value;
    }
    *size = index;
    return array;
}

void write_file(FILE* fp, int* array, size_t size){
    for(int i=0; i<size; i++){
        fprintf(fp, "%d", array[i]);
        if(i<size-1){
            fprintf(fp, ", ");
        }
    }
}
