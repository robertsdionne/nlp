require 'torch'
require 'nn'

local Weights = torch.class('nn.Weights')

function Weights.normalizedInitialization(fan_out, fan_in)
  return (torch.rand(fan_out, fan_in) * 2.0 - 1.0) * math.sqrt(6.0 / (fan_in + fan_out))
end

function Weights.zeros(rows, columns)
  if columns then
    return torch.rand(rows, columns):fill(0)
  else
    return torch.rand(rows):fill(0)
  end
end
