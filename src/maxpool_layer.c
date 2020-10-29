#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
  // Saving our input
  // Probably don't change this
  free_matrix(*l.x);
  *l.x = copy_matrix(in);

  int outw = (l.width-1)/l.stride + 1;
  int outh = (l.height-1)/l.stride + 1;
  matrix out = make_matrix(in.rows, outw*outh*l.channels);

  // TODO: 6.1 - iterate over the input and fill in the output with max values
  int ch, r, row, col, i, j, index;
  int kernel_dist_left, kernel_dist_right;
  kernel_dist_left = 0;
  kernel_dist_right = l.size - 1;
  if (l.size % 2 != 0) {
    // odd
    kernel_dist_left = -l.size / 2;
    kernel_dist_right = l.size / 2;
  }


  for(r = 0; r < in.rows; r++) {
    index = 0;
    // process each image
    for (ch = 0; ch < l.channels; ch++) {
      for (row = 0; row < l.height; row += l.stride) {
        for (col = 0; col < l.width; col += l.stride) {
          float max = -100000000000.0; //FLT_MIN;
          // kernel
          for (i = kernel_dist_left; i <= kernel_dist_right; i++) {  // row
            for (j = kernel_dist_left; j <= kernel_dist_right; j++) {  // col
              float data;
              if (row + i >= 0 && row + i < l.height && col + j >= 0 && col + j < l.width) {
                data = in.data[l.width * row + col + i * l.width + j + ch * l.height * l.width + r * in.cols];
              } else {
                data = 0;
              }
              if (data > max) {
                max = data;
              }
            }
          }
          out.data[r*out.cols + index] = max;
          index++;
        }
      }
    }
  }
  return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
  matrix in    = *l.x;
  matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

  int outw = (l.width-1)/l.stride + 1;
  int outh = (l.height-1)/l.stride + 1;

  // TODO: 6.2 - find the max values in the input again and fill in the
  // corresponding delta with the delta from the output. This should be
  // similar to the forward method in structure.
  int ch, r, row, col, i, j, index;
  int kernel_dist_left, kernel_dist_right;
  kernel_dist_left = 0;
  kernel_dist_right = l.size - 1;
  if (l.size % 2 != 0) {
    // odd
    kernel_dist_left = -l.size / 2;
    kernel_dist_right = l.size / 2;
  }


  for(r = 0; r < in.rows; r++) {
    index = 0;
    // process each image
    for (ch = 0; ch < l.channels; ch++) {
      for (row = 0; row < l.height; row += l.stride) {
        for (col = 0; col < l.width; col += l.stride) {
          float max = -100000000000.0; //FLT_MIN;
          // kernel
          int pos;
          for (i = kernel_dist_left; i <= kernel_dist_right; i++) {  // row
            for (j = kernel_dist_left; j <= kernel_dist_right; j++) {  // col
              float data;
              if (row + i >= 0 && row + i < l.height && col + j >= 0 && col + j < l.width) {
                data = in.data[l.width * row + col + i * l.width + j + ch * l.height * l.width + r * in.cols];
              } else {
                data = 0;
              }
              if (data > max) {
                max = data;
                pos = l.width * row + col + i * l.width + j + ch * l.height * l.width + r * in.cols;
              }
            }
          }
          dx.data[pos] += dy.data[r * dy.cols + index];
          index++;
        }
      }
    }
  }

  return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
  layer l = {0};
  l.width = w;
  l.height = h;
  l.channels = c;
  l.size = size;
  l.stride = stride;
  l.x = calloc(1, sizeof(matrix));
  l.forward  = forward_maxpool_layer;
  l.backward = backward_maxpool_layer;
  l.update   = update_maxpool_layer;
  return l;
}
