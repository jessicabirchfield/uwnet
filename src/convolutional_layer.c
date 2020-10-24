#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_convolutional_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols % b.cols == 0);

    matrix y = copy_matrix(xw);
    int spatial = xw.cols / b.cols;
    int i,j;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            y.data[i*y.cols + j] += b.data[j/spatial];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_convolutional_bias(matrix dy, int n)
{
    assert(dy.cols % n == 0);
    matrix db = make_matrix(1, n);
    int spatial = dy.cols / n;
    int i,j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j/spatial] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{
    int i, j, k;
    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    matrix col = make_matrix(rows, cols);

    // TODO: 5.1
    // Fill in the column matrix with patches from the image
    // int kernel_dist = size / 2;  // distance from center of kernel
    int kernel_dist_left = -size / 2;
    int kernel_dist_right = size / 2;
    if (size % 2 == 0) {
        kernel_dist_left += 1;
    }

    // get seperate channels
    for (int ch = 0; ch < im.c; ch++) {
      for (j = 0; j < im.h; j += stride) {  // rows
        for (i = 0; i < im.w; i += stride) {  // columns
          // -1 to 1 --> -1, 0, 1
          for (int k_row = kernel_dist_left; k_row <= kernel_dist_right; k_row++) {  // going top bottom - vertical
            for (int k_col = kernel_dist_left; k_col <= kernel_dist_right; k_col++) { // going left right - horizontal
              // get all the kernel values and put in output matrix
              int col_row_index = (size * size) * ch + k_row * size + k_col; //+ (size * size) / 2;
              if (size % 2 != 0) {
                col_row_index += (size * size) / 2;
              }
              int col_col_index = (j / stride) * ((im.w - 1) / stride + 1) + (i / stride);

              if (i + k_col < 0 || i + k_col >= im.w || j + k_row < 0 || j + k_row >= im.h) {
                // out of bounds col-wise or row-wise
                // col_row_index = (k_row + kernel_dist) * size + (k_col + kernel_dist);
                // This is correct?
                col.data[col_row_index * cols + col_col_index] = 0;

              } else {
                // this is an actual pixel
                col.data[col_row_index * cols + col_col_index] = get_pixel(im, i + k_col, j + k_row, ch);
              }
            }
          }
        }
      }
    }


    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
image col2im(int width, int height, int channels, matrix col, int size, int stride)
{
    //int i, j, k;

    image im = make_image(width, height, channels);
    int outw = (im.w-1)/stride + 1;
    int rows = im.c*size*size;

    //assert(outw == col.cols);
    assert(rows == col.rows);
    assert(channels == 3);
    assert(col.rows == (size * size * 3));
    // TODO: 5.2
    // Add values into image im from the column matrix
    // for (int ch = 0; ch < channels; ch++) {
        for (int c = 0; c < col.cols; c++) { // columns
          // for (int ch = 0; ch < channels; ch++) {
            for (int r = 0; r < col.rows; r++) { // rows
            // for (int r = size * size * ch; r < size * size * (ch + 1); r++) { // rows
                // get the row r and column c
                if (col.data[r * col.cols + c] != 0) {
                  // STEP 1: use row to calculate which pixel is the center
                  int col_r, k_row, k_col;
                  int ch = r % (size * size);
                  col_r = r - (size * size) * ch;  // shift back to 0-8
                  k_col = (col_r % size) - (size / 2); // k_col
                  k_row = (col_r / size) - (size / 2); // k_row  we know (-1,1) in the kernel

                  // STEP 2: Find position of kernel in the image
                  int im_row, im_col;
                  // outw == width of image taking stride into account (rounding up)
                  im_row = (c % ((height-1)/stride + 1)) * stride;
                  im_col = (c / ((width-1)/stride + 1)) * stride;

                  // STEP 3: Calculate kernel displacement
                  im_row += k_row;
                  im_col += k_col;

                  // STEP 4: Check if this is valid placement
                  if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    assert(ch < 3);
                    // Valid, add value into image
                    float update = get_pixel(im, im_row, im_col, ch) + col.data[r * col.cols + c];
                    set_pixel(im, im_row, im_col, ch, update);
                  }
                }

                // use col to calculate position within kernel

                // if 0 --> ignore
                // else

            //}
        }
    }



    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    assert(in.cols == l.width*l.height*l.channels);
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, j;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.filters);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for(j = 0; j < wx.rows*wx.cols; ++j){
            out.data[i*out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
    matrix y = forward_convolutional_bias(out, l.b);
    free_matrix(out);

    return y;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_convolutional_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    assert(in.cols == l.width*l.height*l.channels);

    int i;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;


    matrix db = backward_convolutional_bias(dy, l.db.cols);
    axpy_matrix(1, db, l.db);
    free_matrix(db);


    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);
    matrix wt = transpose_matrix(l.w);

    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);

        dy.rows = l.filters;
        dy.cols = outw*outh;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(dy, xt);
        axpy_matrix(1, dw, l.dw);

        matrix col = matmul(wt, dy);
        image dxi = col2im(l.width, l.height, l.channels, col, l.size, l.stride);
        memcpy(dx.data + i*dx.cols, dxi.data, dx.cols * sizeof(float));
        free_matrix(col);

        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
        free_image(dxi);

        dy.data = dy.data + dy.rows*dy.cols;
    }
    free_matrix(wt);
    return dx;

}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 5.3
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w  = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
    l.dw = make_matrix(filters, size*size*c);
    l.b  = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
    return l;
}
