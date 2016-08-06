package c2g

import "github.com/Rompei/caffe2go/caffe"

// LayerParameter is alias for LayerParameter.
type LayerParameter interface {
	GetName() string
	GetBottom() []string
	GetTop() []string
	GetLossWeight() []float32
	GetBlobs() []*caffe.BlobProto
	GetInclude() []*caffe.NetStateRule
	GetExclude() []*caffe.NetStateRule
	GetTransformParam() *caffe.TransformationParameter
	GetLossParam() *caffe.LossParameter
	GetAccuracyParam() *caffe.AccuracyParameter
	GetArgmaxParam() *caffe.ArgMaxParameter
	GetConcatParam() *caffe.ConcatParameter
	GetContrastiveLossParam() *caffe.ContrastiveLossParameter
	GetConvolutionParam() *caffe.ConvolutionParameter
	GetDataParam() *caffe.DataParameter
	GetDropoutParam() *caffe.DropoutParameter
	GetDummyDataParam() *caffe.DummyDataParameter
	GetEltwiseParam() *caffe.EltwiseParameter
	GetExpParam() *caffe.ExpParameter
	GetHdf5DataParam() *caffe.HDF5DataParameter
	GetHdf5OutputParam() *caffe.HDF5OutputParameter
	GetHingeLossParam() *caffe.HingeLossParameter
	GetImageDataParam() *caffe.ImageDataParameter
	GetInfogainLossParam() *caffe.InfogainLossParameter
	GetInnerProductParam() *caffe.InnerProductParameter
	GetLrnParam() *caffe.LRNParameter
	GetMemoryDataParam() *caffe.MemoryDataParameter
	GetMvnParam() *caffe.MVNParameter
	GetPoolingParam() *caffe.PoolingParameter
	GetPowerParam() *caffe.PowerParameter
	GetReluParam() *caffe.ReLUParameter
	GetSigmoidParam() *caffe.SigmoidParameter
	GetSoftmaxParam() *caffe.SoftmaxParameter
	GetSliceParam() *caffe.SliceParameter
	GetTanhParam() *caffe.TanHParameter
	GetThresholdParam() *caffe.ThresholdParameter
	GetWindowDataParam() *caffe.WindowDataParameter
}

// Parameter is alias of parameter.
type Parameter interface {
	GetKernelH() uint32
	GetKernelW() uint32
	GetKernelSize() uint32
	GetStrideH() uint32
	GetStrideW() uint32
	GetStride() uint32
	GetPad() uint32
	GetPadH() uint32
	GetPadW() uint32
}

// Blob is alias of Blob.
type Blob interface {
	GetNum() int32
	GetShape() *caffe.BlobShape
	GetHeight() int32
	GetWidth() int32
	GetChannels() int32
}
