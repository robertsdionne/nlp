require "nn"
local leftInSize = 2;
local rightInSize = 3;
local totalInSize = leftInSize + rightInSize;

local leftInput = torch.rand(leftInSize)
local rightInput = torch.rand(rightInSize)
local concatenedInput = torch.rand(leftInSize + rightInSize)
local sumedGradOutput = torch.Tensor(leftInSize):fill(1)

--test CrossCore --random test
dofile "CrossCore.lua"
local core = nn.CrossCore(torch.rand(leftInSize,totalInSize),torch.rand(leftInSize))
core:forward(concatenedInput)
core:backward(concatenedInput,sumedGradOutput)
print(core:parameters())
print(core:getGradWeight())