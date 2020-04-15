package dnn

import (
	"errors"
	"math"
)

//SoftMax performs the softmax calculation
func SoftMax(x, y *Tensor) (err error) {
	if x.nhwc != y.nhwc {
		return errors.New(" x.nhwc != y.nhwc")
	}
	if !comparedims(x.dims, y.dims) {
		return errors.New("x.dims != y.dims")
	}
	dims := make([]int, len(x.dims))
	stride := make([]int, len(x.stride))
	copy(dims, x.dims)
	copy(stride, x.stride)
	if !x.nhwc {
		dims[1] = x.dims[len(x.dims)-1]
		stride[1] = x.stride[len(x.stride)-1]
		for i := 2; i < len(dims); i++ {
			dims[i] = x.dims[i-1]
			stride[i] = x.stride[i-1]
		}
	}
	for i := 0; i < dims[0]; i++ {
		for j := 0; j < dims[1]; j++ {
			for k := 0; k < dims[2]; k++ {
				var denom float64
				for l := 0; l < dims[3]; l++ {
					denom += math.Exp(float64(x.f32data[i*stride[0]+j*stride[1]+k*stride[2]+l]))
				}
				for l := 0; l < dims[3]; l++ {
					y.f32data[i*stride[0]+j*stride[1]+k*stride[2]+l] =
						float32(math.Exp(float64(x.f32data[i*stride[0]+j*stride[1]+k*stride[2]+l])) / denom)
				}
			}
		}
	}
	return nil
}
