package convnetgo

import (
	"errors"
	"fmt"
)

//Convolution contains the parameters that are used to do a convolution
type Convolution struct {
	padding, dilation, stride []int
	set                       bool
	nhwc                      bool
}

//CreateConvolution creates a convolution algo
func CreateConvolution() *Convolution {
	return &Convolution{}
}

//Set sets the convolution settings
func (c *Convolution) Set(padding, stride, dilation []int, NHWC bool) error {
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

//Get gets the convolution settings
func (c *Convolution) Get() (padding, dilation, stride []int) {
	return c.padding, c.dilation, c.stride
}

//FindOutputDims finds the output dims of the convolution
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

//BackwardData does the backward data operation
//dx stores gradients from x in forward propagation. (out)
//w is weights (in)
//dy are gradients stored from layers output (in)
//alpha is for future work it has no function right now
//if beta will multiply the previous values for dx by whatever beta is
func (c *Convolution) BackwardData(dx, w, dy *Tensor, alpha, beta float32) (err error) {

	if len(dx.dims) != len(w.dims) || len(dx.dims) != len(dy.dims) {
		return errors.New("x, w,y need to be the same length")
	}
	if len(dx.dims) < mindimsize || len(dx.dims) > maxdimsize {
		return fmt.Errorf("length of x,w,y need to be between %d and %d", mindimsize, maxdimsize)
	}
	if len(dx.dims)-2 != len(c.stride) {
		return errors.New("convolution stride,dilation,and padding need to be length of x,w,y -2")
	}
	if beta == 0 {
		dx.SetAll(0)
	} else {
		dx.MultAll(beta)
	}
	switch c.nhwc {
	case true:
		switch len(dx.dims) {
		case 4:
			c.backwarddatatNHWC4d(dx, w, dy)
		default:
			return errors.New("Unsupported Number of Tensor Dims")
		}
	case false:
		switch len(dx.dims) {
		case 4:
			c.backwarddataNCHW4d(dx, w, dy)
		default:
			return errors.New("Unsupported Number of Tensor Dims")
		}

	}

	return nil

}

//BackwardFilter updates gradients from dy to dw and db.
//alpha is for future work. beta will multiply previous values of dw by beta before gradient is accumulated
func (c *Convolution) BackwardFilter(x, dw, db, dy *Tensor, alpha, beta float32) (err error) {
	if len(x.dims) != len(dw.dims) || len(x.dims) != len(dy.dims) {
		return errors.New("x, w,y need to be the same length")
	}
	if len(x.dims) < mindimsize || len(x.dims) > maxdimsize {
		return fmt.Errorf("length of x,w,y need to be between %d and %d", mindimsize, maxdimsize)
	}
	if len(x.dims)-2 != len(c.stride) {
		return errors.New("convolution stride,dilation,and padding need to be length of x,w,y -2")
	}
	if beta == 0 {
		dw.SetAll(0)
		db.SetAll(0)
	} else {
		dw.MultAll(beta)
		db.MultAll(beta)
	}
	switch c.nhwc {
	case true:
		switch len(x.dims) {
		case 4:
			c.backwardfilterNHWC4d(x, dw, db, dy)
		default:
			return errors.New("Unsupported Number of Tensor Dims")
		}
	case false:
		switch len(x.dims) {
		case 4:
			c.backwardfilterNCHW4d(x, dw, db, dy)
		default:
			return errors.New("Unsupported Number of Tensor Dims")
		}

	}

	return nil

}

//Forward is the forward propagation.  Calcultions are stored in y
//alpha and beta are for future work. they don't have any function rightnow
func (c *Convolution) Forward(x, w, wb, y *Tensor, alpha, beta float32) (err error) {
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
		switch len(x.dims) {
		case 4:
			c.forwardNHWC4d(x, w, wb, y, alpha, beta)
		default:
			return errors.New("Unsupported Number of Tensor Dims")
		}
	} else {
		switch len(x.dims) {
		case 4:
			c.forwardNCHW4d(x, w, wb, y, alpha, beta)
		default:
			return errors.New("Unsupported Number of Tensor Dims")
		}
	}

	return nil
}
