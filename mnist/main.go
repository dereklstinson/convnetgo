package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/dereklstinson/convnetgo"
)

const mnisttrainimagepath = "/home/ubuntu/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/train-images.idx3-ubyte"
const mnisttrainlabelpath = "/home/ubuntu/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/train-labels.idx1-ubyte"
const mnisttestimagepath = "/home/ubuntu/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/t10k-images.idx3-ubyte"
const mnisttestlabelpath = "/home/ubuntu/go/src/github.com/dereklstinson/GoCuNets/testing/mnist/files/t10k-labels.idx1-ubyte"
const (
	imageMagic = 0x00000803
	labelMagic = 0x00000801
	// Width of the input tensor / picture

)

func main() {
	seeder := rand.New(rand.NewSource(time.Now().Unix()))
	ti, tl, tti, ttl := getmnistdata([]string{mnisttrainimagepath, mnisttrainlabelpath, mnisttestimagepath, mnisttestlabelpath})
	fmt.Println(len(ti), len(tl), len(tti), len(ttl))
	batchsize := 200
	decay1 := float32(.00001)
	decay2 := float32(.001)
	NHWC := true
	var inputlayerweights []int
	var otherlayerdims []int
	var labeldims []int
	var imagedims []int
	var neuronhidden = 12
	bdims := []int{20, 1, 1, 1}
	if NHWC {
		inputlayerweights = []int{neuronhidden, 4, 4, 1}
		otherlayerdims = []int{neuronhidden, 4, 4, neuronhidden}

		labeldims = []int{batchsize, 1, 1, 10}
		imagedims = []int{batchsize, 28, 28, 1}
	} else {
		inputlayerweights = []int{neuronhidden, 1, 4, 4}
		otherlayerdims = []int{neuronhidden, neuronhidden, 4, 4}

		labeldims = []int{batchsize, 10, 1, 1}
		imagedims = []int{batchsize, 1, 28, 28}
	}

	tnbatches := len(ti) / batchsize
	ttbatches := len(tti) / batchsize
	trainimage := make([]*convnetgo.Tensor, tnbatches)
	trainlabel := make([]*convnetgo.Tensor, tnbatches)
	testimage := make([]*convnetgo.Tensor, ttbatches)
	testlabel := make([]*convnetgo.Tensor, ttbatches)
	fmt.Println(len(testlabel))
	fmt.Println(len(trainlabel))
	var err error

	outputtensor, err := convnetgo.CreateTensor(labeldims, NHWC)
	if err != nil {
		panic(err)
	}
	for i := range trainimage {
		trainimage[i], err = convnetgo.CreateTensor(imagedims, NHWC)
		if err != nil {
			panic(err)
		}
		trainlabel[i], err = convnetgo.CreateTensor(labeldims, NHWC)
		if err != nil {
			panic(err)
		}
		err = loadrandomtensor(trainimage[i], trainlabel[i], ti, tl, seeder.Int63())
		if err != nil {
			panic(err)
		}
	}
	for i := range testimage {
		testimage[i], err = convnetgo.CreateTensor(imagedims, NHWC)
		if err != nil {
			panic(err)
		}
		testlabel[i], err = convnetgo.CreateTensor(labeldims, NHWC)
		if err != nil {
			panic(err)
		}
		err = loadrandomtensor(testimage[i], testlabel[i], tti, ttl, seeder.Int63())
		if err != nil {
			panic(err)
		}
	}
	for i := range trainimage {
		trainimage[i].MultAll(float32(1) / float32(255))
		trainimage[i].AddAll(-trainimage[i].Average())
	}
	for i := range testimage {
		testimage[i].MultAll(float32(1) / float32(255))
		testimage[i].AddAll(-testimage[i].Average())
	}
	previousinput := trainimage[0]
	var dpreviousinput *convnetgo.Tensor
	layers := make([]*layer, 0)

	updaters := make([]*trainer, 0)
	wdims := inputlayerweights
	for i := 0; i < 3; i++ {
		layer := createconvlayer(previousinput, dpreviousinput, wdims, bdims, []int{2, 2}, []int{2, 2}, []int{2, 2}, NHWC)
		layers = append(layers, layer)
		updater := createtrainer(layer.w, layer.dw, decay1, decay2)
		updaters = append(updaters, updater)
		updater = createtrainer(layer.b, layer.db, decay1, decay2)
		updaters = append(updaters, updater)
		layer = createactlayer(layer.y, layer.dy, NHWC)
		layers = append(layers, layer)
		previousinput = layer.y
		dpreviousinput = layer.dy
		wdims = otherlayerdims
	}

	classifier := createclassifierlayer(previousinput, dpreviousinput, trainlabel[0], outputtensor, NHWC)
	updater := createtrainer(classifier.w, classifier.dw, decay1, decay2)
	updaters = append(updaters, updater)
	updater = createtrainer(classifier.b, classifier.db, decay1, decay2)
	updaters = append(updaters, updater)
	for i := 0; i < 20; i++ {
		var avgloss float32
		var avgtestloss float32
		var avgpercent float32
		var avgtestpercent float32
		starttime := time.Now()
		for j := range trainimage {
			//	traintimestart := time.Now()

			layers[0].x = trainimage[j]
			classifier.cfdy = trainlabel[j]
			for k := range layers {
				err = layers[k].forward()
				if err != nil {
					panic(err)
				}

			}
			err = classifier.forward()
			if err != nil {
				panic(err)
			}
			//	trainforwardtime := time.Now().Sub(traintimestart).Seconds()
			//	trainbackwarddatatimestart := time.Now()
			err = classifier.backward()
			if err != nil {
				panic(err)
			}
			//fmt.Println(layers[2].dw)
			avgloss += classifier.loss
			avgpercent += classifier.percent
			//	go fmt.Printf("epoch: %v, run: %v, avgbatchloss: %v, avgpercent: %v\n", i, j, classifier.loss, classifier.percent)
			//	fmt.Println(classifier.cfy)
			//	fmt.Println(classifier.loss)
			//	fmt.Println(classifier.cfy)
			for k := len(layers) - 1; k >= 0; k-- {
				err = layers[k].backwarddata()
				if err != nil {
					panic(err)
				}
			}
			var wg sync.WaitGroup
			//	trainbackwarddatatime := time.Now().Sub(trainbackwarddatatimestart).Seconds()
			//	trainbackwardfiltertimestart := time.Now()
			for k := 0; k < len(layers); k++ {
				wg.Add(1)
				go func(k int) {
					err = layers[k].backwardfilter()
					if err != nil {
						panic(err)
					}
					wg.Done()
				}(k)
			}
			wg.Wait()
			//		trainbackwardfiltertime := time.Now().Sub(trainbackwardfiltertimestart).Seconds()
			//	trainupdateweightstimestart := time.Now()
			for k := 0; k < len(updaters); k++ {
				wg.Add(1)
				go func(k int) {
					err = updaters[k].updateweights()
					if err != nil {
						panic(err)
					}
					wg.Done()
				}(k)

			}
			wg.Wait()
			//	trainupdateweightstime := time.Now().Sub(trainupdateweightstimestart).Seconds()
			//	traintotalruntime := time.Now().Sub(traintimestart).Seconds()
			//	go fmt.Printf("epoch: %v, run: %v, avgloss: %v, avgperc: %v, fwt: %v,bwdt: %v,bwdft: %v,updatet: %v, tottime: %v\n",
			//		i, j, classifier.loss, classifier.percent, trainforwardtime, trainbackwarddatatime, trainbackwardfiltertime, trainupdateweightstime, traintotalruntime)

			//	fmt.Println(classifier.dx)
		}

		for j := range testimage {

			layers[0].x = testimage[j]
			classifier.cfdy = testlabel[j]
			for k := range layers {
				err = layers[k].forward()
				if err != nil {
					panic(err)
				}
			}
			err = classifier.forward()
			if err != nil {
				panic(err)
			}
			testloss, testpercent := classifier.testingloss()

			avgtestloss += testloss
			avgtestpercent += testpercent

		}

		fmt.Printf("trainloss: %v,trainpercent: %v ,testloss: %v,testpercent: %v, epochtime: %v\n", avgloss/float32(len(trainimage)), avgpercent/float32(len(trainimage)), avgtestloss/(float32)(len(testimage)), avgtestpercent/(float32)(len(testimage)), time.Now().Sub(starttime).Seconds())

	}

}

type trainer struct {
	w, dw, gsum, xsum *convnetgo.Tensor
	a                 *convnetgo.Adam
	l1, l2            float32
	d1, d2            float32
}
type classifierlayer struct {
	alpha, beta      float32
	x, dx, nny, nndy *convnetgo.Tensor
	w, dw, b, db     *convnetgo.Tensor
	cfy, cfdy        *convnetgo.Tensor
	loss, percent    float32
}

func createclassifierlayer(x, dx, y, dy *convnetgo.Tensor, NHWC bool) *classifierlayer {
	var err error
	cl := new(classifierlayer)
	cl.x = x
	cl.cfy = y
	cl.cfdy = dy
	cl.dx = dx
	cl.nndy, err = y.ZeroClone()
	if err != nil {
		panic(err)
	}
	cl.nny, err = y.ZeroClone()
	if err != nil {
		panic(err)
	}

	xdims := x.Dims()

	nweights := 1
	for i := 1; i < len(xdims); i++ {
		nweights *= xdims[i]
	}
	ydims := y.Dims()

	wdims := make([]int, len(xdims))
	bdims := make([]int, len(xdims))
	for i := range wdims {
		wdims[i] = 1
		bdims[i] = 1
	}
	if NHWC {
		channels := ydims[len(ydims)-1]
		wdims[0] = channels
		wdims[1] = nweights
		bdims[0] = channels
	} else {
		channels := ydims[1]
		wdims[0] = channels
		wdims[len(wdims)-1] = nweights
		bdims[0] = channels
	}
	cl.w, err = convnetgo.CreateRandomizedWeightsTensor(wdims, xdims, NHWC)
	if err != nil {
		panic(err)
	}
	cl.dw, err = convnetgo.CreateTensor(wdims, NHWC)
	if err != nil {
		panic(err)
	}
	cl.b, err = convnetgo.CreateTensor(bdims, NHWC)
	if err != nil {
		panic(err)
	}
	cl.db, err = convnetgo.CreateTensor(bdims, NHWC)
	if err != nil {
		panic(err)
	}

	return cl
}
func (c *classifierlayer) forward() error {
	err := convnetgo.FullyConnectedForward(c.x, c.w, c.b, c.nny, c.alpha, c.beta)
	if err != nil {
		return err
	}
	return convnetgo.SoftMax(c.nny, c.cfy, c.alpha, c.beta)
}
func (c *classifierlayer) testingloss() (loss, percent float32) {
	return convnetgo.SoftMaxLossandPercent(c.cfdy, c.cfy)
}
func (c *classifierlayer) backward() error {
	c.loss, c.percent = convnetgo.SoftMaxLossandPercent(c.cfdy, c.cfy)
	err := c.nndy.Add(c.cfy, c.cfdy, 1, -1, 0)
	if err != nil {
		return err
	}
	err = convnetgo.FullyConnectedBackwardData(c.dx, c.w, c.nndy, c.alpha, c.beta)
	if err != nil {
		return err
	}
	err = convnetgo.FullyConnectedBackwardFilter(c.x, c.dw, c.db, c.nndy, c.alpha, c.beta)
	return nil
}
func (t *trainer) updateweights() error {
	t.l1, t.l2 = convnetgo.L1L2Regularization(t.d1, t.d2, t.dw, t.w)
	return t.a.UpdateWeights(t.gsum, t.xsum, t.dw, t.w, true)
}
func (t *trainer) l1l2loss() (l1, l2 float32) {
	return t.l1, t.l2
}
func createtrainer(w, dw *convnetgo.Tensor, d1, d2 float32) *trainer {
	var err error
	t := new(trainer)
	t.w = w
	t.dw = dw
	t.d1 = d1
	t.d2 = d2
	t.gsum, err = w.ZeroClone()
	if err != nil {
		panic(err)
	}
	t.xsum, err = w.ZeroClone()
	if err != nil {
		panic(err)
	}
	t.a = convnetgo.CreateAdamTrainer(nil)
	return t
}
func createconvlayer(x, dx *convnetgo.Tensor, wdims, bdims, pad, stride, dilation []int, nhwc bool) *layer {
	var err error
	l := new(layer)
	l.x = x
	l.dx = dx
	l.alpha = 1
	l.beta = 0
	l.b, err = convnetgo.CreateTensor(bdims, nhwc)
	if err != nil {
		panic(err)
	}
	l.db, err = convnetgo.CreateTensor(bdims, nhwc)
	if err != nil {
		panic(err)
	}
	l.dw, err = convnetgo.CreateTensor(wdims, nhwc)
	if err != nil {
		panic(err)
	}
	l.w, err = convnetgo.CreateRandomizedWeightsTensor(wdims, x.Dims(), nhwc)
	if err != nil {
		panic(err)
	}
	l.conv = convnetgo.CreateConvolution()
	err = l.conv.Set(pad, stride, dilation, nhwc)
	if err != nil {
		panic(err)
	}
	outdims := l.conv.FindOutputDims(x, l.w)
	l.y, err = convnetgo.CreateTensor(outdims, nhwc)
	if err != nil {

		panic(err)
	}
	l.dy, err = convnetgo.CreateTensor(outdims, nhwc)
	if err != nil {
		panic(err)
	}
	return l
}
func createactlayer(x, dx *convnetgo.Tensor, nhwc bool) *layer {
	var err error
	l := new(layer)
	l.alpha = 1
	l.beta = 0
	l.x = x
	l.dx = dx
	l.y, err = convnetgo.CreateTensor(x.Dims(), nhwc)
	if err != nil {
		panic(err)
	}
	l.dy, err = convnetgo.CreateTensor(x.Dims(), nhwc)
	if err != nil {
		panic(err)
	}
	l.act, err = convnetgo.CreateLeakyRelu(.01, 1)
	if err != nil {
		panic(err)

	}
	return l

}
func (l *layer) forward() error {
	if l.conv != nil {
		return l.conv.Forward(l.x, l.w, l.b, l.y, l.alpha, l.beta)

	}
	if l.act != nil {
		return l.act.Forward(l.x, l.y, l.alpha, l.beta)
	}
	return errors.New("Please set layer")

}
func (l *layer) backwardfilter() error {
	if l.conv != nil {
		return l.conv.BackwardFilter(l.x, l.dw, l.db, l.dy, l.alpha, l.beta)
	}
	return nil
}
func (l *layer) backwarddata() error {
	if l.conv != nil {
		if l.dx != nil {
			err := l.conv.BackwardData(l.dx, l.w, l.dy, l.alpha, l.beta)
			if err != nil {
				return err
			}
		}
		return nil
	}
	if l.act != nil {
		return l.act.Backward(l.x, l.dx, l.dy, l.alpha, l.beta)
	}
	return errors.New("Please set layer")
}

type layer struct {
	alpha, beta  float32
	x, dx, y, dy *convnetgo.Tensor
	w, dw, b, db *convnetgo.Tensor
	conv         *convnetgo.Convolution
	act          *convnetgo.LeakyRelu
}

func loadrandomtensor(ti, tl *convnetgo.Tensor, data, label [][]float32, seed int64) error {
	rng := rand.New(rand.NewSource(seed))

	batch := ti.Dims()[0]
	ndata := len(data)

	var err error
	batchdata := make([]float32, 0)
	batchlabel := make([]float32, 0)
	for i := 0; i < batch; i++ {
		rngindex := rng.Int() % ndata
		batchdata = append(batchdata, data[rngindex]...)
		batchlabel = append(batchlabel, label[rngindex]...)
	}
	err = ti.LoadFromSlice(batchdata)
	if err != nil {
		return err
	}
	err = tl.LoadFromSlice(batchlabel)
	if err != nil {
		return err
	}
	return nil
}
func readImageFile(r io.Reader) ([][]float32, error) {

	var err error

	var (
		magic int32
		n     int32
		nrow  int32
		ncol  int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {
		fmt.Println("readImage... magic,imageMagic,labelMagic", magic, imageMagic, labelMagic)
		return nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	if err = binary.Read(r, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	imgflts := make([][]float32, n)
	imgs := make([][]byte, n)
	size := int(nrow * ncol)
	for i := 0; i < int(n); i++ {
		imgflts[i] = make([]float32, size)
		imgs[i] = make([]byte, size)

		actual, err := io.ReadFull(r, imgs[i])
		if err != nil {
			return nil, err
		}
		if size != actual {
			return nil, os.ErrInvalid
		}
		for j := 0; j < size; j++ {
			imgflts[i][j] = float32(imgs[i][j])
		}
	}

	return imgflts, nil
}
func readLabelFile(r io.Reader) ([][]float32, []int, error) {
	var err error

	var (
		magic int32
		n     int32
	)
	if err = binary.Read(r, binary.BigEndian, &magic); err != nil {

		return nil, nil, err
	}
	if magic != labelMagic {
		fmt.Println("readLabel... magic,imageMagic,labelMagic", magic, imageMagic, labelMagic)
		return nil, nil, os.ErrInvalid
	}
	if err = binary.Read(r, binary.BigEndian, &n); err != nil {

		return nil, nil, err
	}
	labels := make([][]float32, n)
	numbers := make([]int, n)
	for i := 0; i < int(n); i++ {
		var l uint8
		if err := binary.Read(r, binary.BigEndian, &l); err != nil {
			return nil, nil, err
		}
		numbers[i] = int(l)
		labels[i] = append(labels[i], makeonehotstate(l)...)
	}
	return labels, numbers, nil
}
func getmnistdata(mnistpath []string) (traindata, trainlabel, testdata, testlabel [][]float32) {
	mniststuff := make([][][]float32, 4)
	mnisturls := []string{"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"}

	for i := range mniststuff {
		file, err := os.Open(mnistpath[i])
		if err != nil {
			panic(err)
		}

		resp, err := http.Get(mnisturls[i])
		if err != nil {
			panic(err)
		}
		defer resp.Body.Close()

		mniststuff[i], err = readImageFile(file)
		if err != nil {

			file.Close()
			file2, err2 := os.Open(mnistpath[i])
			defer file2.Close()
			if err2 != nil {
				panic(err2)
			}
			mniststuff[i], _, err2 = readLabelFile(file2)
			if err2 != nil {
				panic(err2)
			}
			fmt.Println("LABEL")
		} else {
			file.Close()
			fmt.Println("IMAGE")
		}
		fmt.Println(len(mniststuff[i]))
	}
	traindata, trainlabel, testdata, testlabel = mniststuff[0], mniststuff[1], mniststuff[2], mniststuff[3]

	return traindata, trainlabel, testdata, testlabel
}

func makeonehotstate(input uint8) []float32 {
	x := make([]float32, 10)
	x[input] = float32(1.0)
	return x

}
