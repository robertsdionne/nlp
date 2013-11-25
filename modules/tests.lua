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

--test CrossWord
dofile "CrossWord.lua"
local wordSize = rightInSize;
local wordIndexSize = 1;
local inputWord = torch.rand(wordSize);
local inputIndex = torch.rand(wordIndexSize);
local crossWord = nn.CrossWord(inputWord, inputIndex);
print("getGradWeight:");
print(crossWord:getGradWeight(inputWord));
print("getOutput():");
print(crossWord:getOutput());

--test CrossTag
dofile "CrossTag.lua"
local featureSize = leftInSize;
local classesSize = 3;
local weight = torch.rand(classesSize,featureSize);
local bias = torch.rand(classesSize);
local tag = 2;
local crossTag = nn.CrossTag(weight, bias, tag);
local nodeRepresentation = torch.rand(leftInSize);
crossTag:forwardBackward(nodeRepresentation);
print("getGradWeight():");
print(crossTag:getGradWeight());
print("getGradInput():");
print(crossTag:getGradInput());