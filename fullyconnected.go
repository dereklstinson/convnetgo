package dnn

import (
	"errors"
	"sync"
)

//FullyConnectedForward does the forward operation for a fully connected neural network
//x is input tensor (in)
//w is weight tensor (in)
//b is bias tensor (in)
//y is the output tensor (out)
//
//Batches are performed in parrallel
func FullyConnectedForward(x, w, b, y *Tensor) error {
	nvol := findvolume(w.dims[1:])
	xvol := findvolume(x.dims[1:])

	if nvol != xvol {
		return errors.New("neuron feature map vol != batch volume of input")
	}

	neurons := w.dims[0]
	yvol := findvolume(y.dims[1:])
	if yvol != neurons {
		return errors.New("Neuron outputs not matching y tensors volume")
	}
	bvol := findvolume(b.dims)
	if bvol != neurons {
		return errors.New("Bias size != to output/neuron size")
	}
	batchsize := x.dims[0]
	batchstride := x.stride[0]

	var wg sync.WaitGroup
	for i := 0; i < batchsize; i++ {
		wg.Add(1)
		go func(i int) {
			xyoffset := batchstride * i
			for j := 0; j < neurons; j++ {
				neuronoffset := w.stride[0] * j
				var adder float32
				for k := 0; k < xvol; k++ {
					adder += w.f32data[neuronoffset+k] * x.f32data[xyoffset+k]
				}

				y.f32data[xyoffset+j] = adder + b.f32data[neuronoffset]
			}
			wg.Done()
		}(i)

	}
	wg.Wait()
	return nil
}

//FullyConnectedBackwardData does the backward data operation for a fully connected neural network
//dx stores  gradients found for the input of this layer (out)
//w is weight tensor (in)
//dy is are the gradients found for the output of this layer (out)
//
//Batches are done in parallel
func FullyConnectedBackwardData(dx, w, dy *Tensor) error {
	nvol := findvolume(w.dims[1:])  //nvol is the size of the weights of each neuron
	xvol := findvolume(dx.dims[1:]) //
	if nvol != xvol {
		return errors.New("FullyConnectedBackwardData - neuron feature map vol != batch volume of dx")
	}
	neurons := w.dims[0]
	yvol := findvolume(dy.dims[1:])
	if yvol != neurons {
		return errors.New("Neuron outputs not matching y tensors volume")
	}

	batchsize := dx.dims[0]
	batchstride := dx.stride[0]
	var wg sync.WaitGroup
	for i := 0; i < batchsize; i++ {
		wg.Add(1)
		go func(i int) {
			xyoffset := batchstride * i
			for j := 0; j < neurons; j++ {
				neuronoffset := w.stride[0] * j
				var grad = dy.f32data[xyoffset+j]
				for k := 0; k < xvol; k++ {
					dx.f32data[xyoffset+k] += grad * w.f32data[neuronoffset+k]
				}
			}
			wg.Done()
		}(i)

	}
	wg.Wait()
	return nil

}

//FullyConnectedBackwardFilter does the backward filter operation for a fully connected neural network
//x is the input for this layer (in)
//dw stores the gradients for the weight tensor (output)
//db stores the gradients for the bias (output)
//dy is are the gradients found for the output of this layer (out)
//
//Batches are done in parallel
func FullyConnectedBackwardFilter(x, dw, db, dy *Tensor) error {
	nvol := findvolume(dw.dims[1:])
	xvol := findvolume(x.dims[1:])
	if nvol != xvol {
		return errors.New("neuron feature map vol != batch volume of input")
	}
	neurons := dw.dims[0]
	yvol := findvolume(dy.dims[1:])
	if yvol != neurons {
		return errors.New("Neuron outputs not matching y tensors volume")
	}
	bvol := findvolume(db.dims)
	if bvol != neurons {
		return errors.New("Bias size != to output/neuron size")
	}
	batchsize := x.dims[0]
	batchstride := x.stride[0]

	var wg sync.WaitGroup
	for i := 0; i < batchsize; i++ {
		wg.Add(1)
		go func(i int) {
			xyoffset := batchstride * i
			for j := 0; j < neurons; j++ {
				neuronoffset := dw.stride[0] * j
				var grad = dy.f32data[xyoffset+j]
				for k := 0; k < xvol; k++ {
					dw.f32data[neuronoffset+k] += x.f32data[xyoffset+k] * grad

				}
				db.f32data[neuronoffset] += grad
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	return nil
}
