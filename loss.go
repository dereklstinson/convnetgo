package dnn

import (
	"errors"
	"math"
)

func SoftMax(x, y *Tensor) (err error) {
	if x.nhwc != y.nhwc {
		return errors.New(" x.nhwc != y.nhwc")
	}
	if !comparedims(x.dims, y.dims) {
		return errors.New("x.dims != y.dims")
	}
	if x.nhwc {
		for i := 0; i < x.dims[0]; i++ {
			for j := 0; j < x.dims[1]; j++ {
				for k := 0; k < x.dims[2]; k++ {
					var denom float64
					for l := 0; l < x.dims[3]; l++ {
						denom += math.Exp(float64(x.f32data[i*x.stride[0]+j*x.stride[1]+k*x.stride[2]+l]))
					}
					for l := 0; l < x.dims[3]; l++ {
						y.f32data[i*x.stride[0]+j*x.stride[1]+k*x.stride[2]+l] =
							float32(math.Exp(float64(x.f32data[i*x.stride[0]+j*x.stride[1]+k*x.stride[2]+l])) / denom)
					}
				}
			}
		}
	}

	return nil
}
