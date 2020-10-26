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
    for(r = 0; r < in.rows; r++) {
      index = 0;
      // process each image
      for (ch = 0; ch < l.channels; ch++) {  
	for (row = 0; row < l.height; row += l.stride) {
          for (col = 0; col < l.width; col += l.stride) {
            float max = FLT_MIN;
            // kernel
            for (i = 0; i < l.size; i++) {
              for (j = 0; j < l.size; j++) {
                float data = in.data[l.width * row + col + i * l.width + j + ch * l.height * l.width];
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

        // //image im = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        // // image out_img = make_image(outw, outh, l.channels);
        //
        // //matrix x = im2col(im, l.size, l.stride);
        // //rows = x.rows / l.channels;
        // rows = l.size * l.size;
        // // Iterate through the columns
        // assert(l.channels == 3);
        // for (ch = 0; ch < l.channels; ch++) {
        //     float max = FLT_MIN;
        //     for (c = 0; c < x.cols; c++) {
        //         // Iterate through the rows
        //         for (r = ch * rows; r < rows * (ch + 1); r++) {
        //             float data = x.data[r * x.cols + c];
        //             if (data > max) {
        //                 max = data;
        //             }
        //         }
        //         // set_pixel(out_img, c, r, ch, max);
        //         // im.data[x + im.w*(y + im.h*c)];
        //          out.data[i*out.cols + c] = max;
        //         // out.data[c + out.rows * (i * out.cols * ch)] = max;
        //         // out.data[(i * out.cols) + c + outw * (r + outh * l.channels)] = max;
        //     }
        // }



        // for (ch = 0; ch < im.c; ch++) {
        //     for (j = 0; j < example.h; j += l.stride) {  // rows
        //         for (i = 0; i < example.w; i += l.stride) {  // columns
        //             int max = FLT_MIN;
        //             for (int k_row = kernel_dist_left; k_row <= kernel_dist_right; k_row++) {  // going top bottom - vertical
        //                 for (int k_col = kernel_dist_left; k_col <= kernel_dist_right; k_col++) { // going left right - horizontal
        //                     int col_row_index = (l.size * l.size) * ch + k_row * l.size + k_col; //+ (size * size) / 2;
        //                     if (l.size % 2 != 0) {
        //                       col_row_index += (l.size * l.size) / 2;
        //                     }
        //                     int col_col_index = (j / stride) * ((im.w - 1) / stride + 1) + (i / stride);
        //                 }
        //             }
        //         }
        //     }
        // }
        // Iterate through the columns for each channel

        // matrix wx = matmul(l.w, x);
        // for(j = 0; j < wx.rows*wx.cols; ++j){
        //     out.data[i*out.cols + j] = //wx.data[j];
        // }
        // free_matrix(x);
        // free_matrix(wx);
  //  }

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
