package convnetgo

import (
	"testing"
)

func TestCreateRandomizedWeightsTensor(t *testing.T) {
	tensor, err := CreateRandomizedWeightsTensor([]int{4, 4, 4, 8}, []int{4, 20, 20, 8}, false)
	if err != nil {
		t.Error(err)
	}
	t.Error(tensor.f32data)
}
