package convnetgo

import (
	"errors"
	"math"
)

//SoftMax performs the softmax calculation
//alpha and beta have no function right now
func SoftMax(x, y *Tensor, alpha, beta float32) (err error) {
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

		dims[len(x.dims)-1] = x.dims[1]
		stride[len(x.stride)-1] = x.stride[1]
		for i := 1; i < len(dims)-1; i++ {
			dims[i] = x.dims[i+1]
			stride[i] = x.stride[i+1]
		}
	}
	for i := 0; i < dims[0]; i++ {
		for j := 0; j < dims[1]; j++ {
			for k := 0; k < dims[2]; k++ {
				var denom float64
				for l := 0; l < dims[3]; l++ {
					denom += math.Exp(float64(x.f32data[i*stride[0]+j*stride[1]+k*stride[2]+stride[3]*l]))
				}
				for l := 0; l < dims[3]; l++ {
					y.f32data[i*stride[0]+j*stride[1]+k*stride[2]+l] =
						float32(math.Exp(float64(x.f32data[i*stride[0]+j*stride[1]+k*stride[2]+stride[3]*l])) / denom)
				}
			}
		}
	}
	return nil
}

//SoftMaxLossandPercent will do the softmax loss of the layer.  It will be an average of all the indicators(anything greater than 0).
//alpha and beta have no function right now
func SoftMaxLossandPercent(target, output *Tensor) (avgloss float32, avgpercent float32) {
	var lossadder float64
	var val float32
	var counter int
	var percentadder float32
	for i := range target.f32data {
		if target.f32data[i] > 0 {
			val = output.f32data[i]
			percentadder += val
			lossadder += -math.Log(float64(val))
			counter++
		}
	}
	return float32(lossadder) / float32(counter), percentadder / float32(counter)
}
