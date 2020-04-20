package convnetgo

import (
	"errors"
	"math"
)

//L1L2Regularization performs the regularizaion on dw. This should be ran before a trainer like adam or momentum
func L1L2Regularization(decay1, decay2 float32, dw, w *Tensor) (l1, l2 float32) {
	batch := (float32)(dw.dims[0])
	l1, l2 = l1l2Regularization(decay1, decay2, batch, w.f32data, dw.f32data)
	return l1, l2
}

//Adam is the adam trainer
type Adam struct {
	rate, eps, beta1, beta2, decay1, decay2, l1, l2 float32
	counter                                         uint64
}

const defaultadambeta1 = (float32)(0.9)
const defaultadambeta2 = (float32)(0.999)
const defaultadameps = (float32)(1e-8)
const defaultadamrate = (float32)(.001)

//AdamOptions are options that can be passed to CreateAdamTrainer
type AdamOptions struct {
	Rate   float32
	Beta1  float32
	Beta2  float32
	Eps    float32
	Decay1 float32
	Decay2 float32
}

//CreateAdamTrainer creates an adam trainer.  If options is nil then default values will be used.
func CreateAdamTrainer(options *AdamOptions) *Adam {
	if options == nil {
		return &Adam{
			rate:   defaultadamrate,
			beta1:  defaultadambeta1,
			beta2:  defaultadambeta2,
			eps:    defaultadameps,
			decay1: 0,
			decay2: .0001,
		}
	}
	return &Adam{
		rate:   options.Rate,
		beta1:  options.Beta1,
		beta2:  options.Beta2,
		eps:    options.Eps,
		decay1: options.Decay1,
		decay2: options.Decay2,
	}
}

//UpdateWeights updates the weights of w
//
//dw is the accumulated gradients for the weights
//
//gsum,xsum are accumulators used to smooth out the training.  Should be the same size as w and dw.
func (a *Adam) UpdateWeights(gsum, xsum, dw, w *Tensor, multithreaded bool) error {
	if len(gsum.f32data) != len(xsum.f32data) || len(gsum.f32data) != len(dw.f32data) || len(gsum.f32data) != len(w.f32data) {
		return errors.New("(a *Adam) UpdateWeights: all tensors passed need to have the data size equal")
	}

	a.counter++
	denomb1 := 1.0 - (float32)(math.Pow(float64(a.beta1), float64(a.counter)))
	denomb2 := 1.0 - (float32)(math.Pow(float64(a.beta2), float64(a.counter)))
	/*
		if multithreaded{
			nneurons:=dw.dims[0]
			neuronstride:=dw.stride[0]
			neuronelements:= findvolume(dw.dims[1:])
			for i:=0;i<nneurons;i++{
				neuronoffset:=i*neuronstride
				go func(neuronoffset, neuronelements int,denomb1,denomb2 float32){
					for j:=0;j<neuronelements;j++{
						gsum.f32data[neuronoffset+j] = (a.beta1 * gsum.f32data[neuronoffset+j]) + ((1.0 - a.beta1) * dw.f32data[neuronoffset+j])
						xsum.f32data[neuronoffset+j] = (a.beta2 * xsum.f32data[neuronoffset+j]) + ((1.0 - a.beta2) * dw.f32data[neuronoffset+j] * dw.f32data[neuronoffset+j])
						w.f32data[neuronoffset+j] += -((gsum.f32data[neuronoffset+j] * a.rate) / denomb1) /
							((float32)(math.Sqrt((float64)(xsum.f32data[neuronoffset+j]/denomb2))) + a.eps)
					}
				}(neuronoffset,neuronelements,denomb1,denomb2)
			}
			return nil
		}*/

	for i := range dw.f32data {

		gsum.f32data[i] = (a.beta1 * gsum.f32data[i]) + ((1.0 - a.beta1) * dw.f32data[i])
		gsumt := gsum.f32data[i] / denomb1
		xsum.f32data[i] = (a.beta2 * xsum.f32data[i]) + ((1.0 - a.beta2) * dw.f32data[i] * dw.f32data[i])
		xsumt := xsum.f32data[i] / denomb2
		w.f32data[i] += -(a.rate * gsumt) / ((float32)(math.Sqrt((float64)(xsumt))) + a.eps)

	}

	return nil
}

func l1l2Regularization(decay1, decay2, batch float32, w, dw []float32) (l1, l2 float32) {
	var grad1, grad2 float32
	for i := range w {
		l1 += abs(w[i]) * decay1
		l2 += (w[i] * w[i] * decay2) / 2

		if w[i] > 0 {
			grad1 = decay1
		} else {
			grad1 = -decay1
		}
		grad2 = w[i] * decay2

		dw[i] = (dw[i] + grad1 + grad2) / batch
	}
	return l1, l2
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
