package dnn

import (
	"errors"
	"fmt"
)

type Convolution struct {
	padding, dilation, stride []int
	set                       bool
	nhwc                      bool
}

func CreateConvolution() *Convolution {
	return &Convolution{}
}
func (c *Convolution) Set(padding, dilation, stride []int, NHWC bool) error {
	if len(padding) != len(dilation) || len(padding) != len(stride) {
		return errors.New("length of padding dilation and stride need to be the same")
	} else if len(padding) < mindimsize-2 || len(padding) > maxdimsize-2 {
		return fmt.Errorf("len of padding dilation and stride need to between %d and %d", mindimsize-2, maxdimsize-2)
	}
	c.nhwc = NHWC
	c.padding = make([]int, len(padding))
	c.dilation = make([]int, len(padding))
	c.stride = make([]int, len(padding))
	copy(c.padding, padding)
	copy(c.dilation, dilation)
	copy(c.stride, stride)
	c.set = true
	return nil
}
func (c *Convolution) Get() (padding, dilation, stride []int) {
	return c.padding, c.dilation, c.stride
}
func (c *Convolution) FindOutputDims(x, w *Tensor) []int {
	output := make([]int, len(x.dims))
	if c.nhwc {
		output[0] = x.dims[0]
		for i := 0; i < len(c.padding); i++ {
			off := i + 1
			output[off] = findoutputdim(x.dims[off], w.dims[off], c.stride[i], c.padding[i], c.dilation[i])
		}
		output[len(output)-1] = w.dims[0]
		return output
	}
	output[0] = x.dims[0]
	output[1] = w.dims[0]
	for i := 0; i < len(c.padding); i++ {
		off := i + 2
		output[off] = findoutputdim(x.dims[off], w.dims[off], c.stride[i], c.padding[i], c.dilation[i])
	}

	return output

}
func findoutputdim(x, w, s, p, d int) int {
	y := x + (2 * p) - (((w - 1) * d) + 1)
	if y < 0 {
		return -1
	}
	return divideup(y, s) + 1
}
func divideup(num, den int) int {
	if num%den != 0 {
		return (num / den) + 1
	}
	return num / den
}

func (c *Convolution) Forward(x, w, wb, y *Tensor) (err error) {
	if len(x.dims) != len(w.dims) || len(x.dims) != len(y.dims) {
		return errors.New("x, w,y need to be the same length")
	}
	if len(x.dims) < mindimsize || len(x.dims) > maxdimsize {
		return fmt.Errorf("length of x,w,y need to be between %d and %d", mindimsize, maxdimsize)
	}
	if len(x.dims)-2 != len(c.stride) {
		return errors.New("convolution stride,dilation,and padding need to be length of x,w,y -2")
	}
	if c.nhwc {

	} else {
		switch len(x.dims) {
		case 4:
			c.forwardNCHW4d(x, w, wb, y)
		case 5:
			c.forwardNCHW5d(x, w, wb, y)
		case 6:
			c.forwardNCHW6d(x, w, wb, y)
		case 7:
			c.forwardNCHW7d(x, w, wb, y)
		case 8:
			c.forwardNCHW8d(x, w, wb, y)
		}
	}

	return nil
}