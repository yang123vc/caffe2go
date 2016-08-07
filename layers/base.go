package layers

import (
	"errors"
)

// BaseLayer is base struct of Layers.
type BaseLayer struct {
	Name string
	Type string
}

// NewBaseLayer is constructor.
func NewBaseLayer(name, t string) *BaseLayer {
	return &BaseLayer{
		Name: name,
		Type: t,
	}
}

// GetName is method to return name of the layer.
func (b *BaseLayer) GetName() string {
	return b.Name
}

// GetType is method to return type of the layer.
func (b *BaseLayer) GetType() string {
	return b.Type
}

// Forward is base function of Forward.
func (b *BaseLayer) Forward(input [][][]float32) ([][][]float32, error) {
	return input, errors.New("Forward is not implemented.")
}
