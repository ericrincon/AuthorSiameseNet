require "nn"
require 'cutorch'

local model = {}

function model.buildModel()

  -- Directly from the torch7 docs
  -- inputFrameSize: The input frame size expected in sequences given into
  -- forward().
  -- outputFrameSize: The output frame size the convolution layer will produce.
  -- kW: The kernel width of the convolution
  -- dW: The step of the convolution. Default is 1.


  -- Use a parallel table to create the siamese network
  siameseNetwork = nn.ParallelTable()

  -- Create a siamese node
  nodeLeft = nn.Sequential()
  nodeLeft:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW))
  nodeLeft:add(RelU())
  nodeLeft:add(nn.TemporalMaxPooling(kW))

  -- Create another siamese node
  nodeRight = nn.Sequential()
  nodeRight:add(nn.TemporalConvolution(inputFrameSize, outputFrameSize, kW))
  nodeRight:add(nn.RelU())
  nodeRight:add(nn.TemporalMaxPooling(kW)(activationRight))

  siameseNetwork:add(maxPoolingLeft)
  siameseNetwork:add(maxPoolingRight)

  siameseNetwork:add(nn.Softmax())

  return siameseNetwork
end

return model
