package convnetgo

import "sync"

//this still needs work
func im2col(t *Tensor, wspatial, pad, stride, dilation []int) (x *Tensor) {
	tdims := t.Dims()
	//	tstride := t.Stride()
	tbatchvol := findvolume(tdims[1:])
	tbatches := tdims[0]
	tspacialdims := findspatialdims(t)
	osdims := findoutputspacialdims(tspacialdims, wspatial, pad, stride, dilation)
	im2cbatchvol := findvolume(osdims) * findvolume(tspacialdims) * getchannellength(t)
	im2cdata := make([]float32, findvolume(osdims)*findvolume(tspacialdims)*getchannellength(t)*tbatches)

	var wg sync.WaitGroup
	for i := 0; i < tbatches-1; i++ {
		tbatchdata := t.f32data[i*tbatchvol : (i+1)*tbatchvol]
		im2cbatchdata := im2cdata[i*im2cbatchvol : (i+1)*im2cbatchvol]
		wg.Add(1)
		go func(tbatchdata, im2cbatchdata []float32, i int) {

			var counter int

			sh := -pad[0]                                              //initiate zero padding h
			for yh := 0; yh < t.dims[1]; yh, sh = yh+1, sh+stride[0] { //output dims1
				sw := -pad[1]                                              //initialize zero padding w
				for yw := 0; yw < t.dims[2]; yw, sw = yw+1, sw+stride[1] { //output dim2
					for wh := 0; wh < wspatial[0]; wh++ {
						xh := sh + (wh * dilation[0])  // This tells the height position on the x tensor
						if xh >= 0 && xh < t.dims[1] { //this checks if it is in bounds of the x tensor
							for ww := 0; ww < wspatial[1]; ww++ {
								xw := sw + (ww * dilation[1])  //This is the width position on the x tensor
								if xw >= 0 && xw < t.dims[2] { //check if xw is in bounds with the x tensor
									for xc := 0; xc < t.dims[3]; xc++ {
										im2cbatchdata[counter] = tbatchdata[(t.stride[1]*xh)+(t.stride[2]*xw)+(t.stride[3]*xc)]
										counter++
									}

								} else {
									im2cbatchdata[counter] = 0
									counter++
								}
							}
						} else {
							im2cbatchdata[counter] = 0
							counter++
						}
					}
				}
			}

			wg.Done()
		}(tbatchdata, im2cbatchdata, i)

	}
	wg.Wait()
	return nil
}
