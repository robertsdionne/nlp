require "nn"
local leftInSize = 2;
local rightInSize = 3;
local totalInSize = leftInSize + rightInSize;

local leftInput = torch.rand(leftInSize)
local rightInput = torch.rand(rightInSize)
local gradOutput = torch.rand(leftInSize)
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
crossTag:forward(nodeRepresentation);
crossTag:backward(nodeRepresentation);
print("getGradWeight():");
print(crossTag:getGradWeight());
print("getGradInput():");
print(crossTag:getGradInput());
print("getPredTag():");
print(crossTag:getPredTag());

--test Cross
dofile "Cross.lua"
local testCross = nn.Cross(core,crossWord,crossTag)
print(testCross:forward(leftInput))
print(testCross:backward(leftInput,gradOutput))
print(testCross:getGradParameters())

--now test CrossRNN
dofile "CrossRNN.lua"
local lookUpTable = torch.rand(3);		--lookUpTable not used in RNN.
local initialNode = torch.rand(leftInSize);
local testCrossRNN = nn.CrossRNN(leftInSize, rightInSize, classesSize, lookUpTable, initialNode);
--create a simple sentenceTuple
local sentence = {torch.rand(rightInSize),torch.rand(rightInSize),torch.rand(rightInSize)};
local index = {1,2,3};
local tag = {1,2,3};
local sentenceTuple = {sentence = sentence, index = index, tag = tag};
local learningRates = 0.1;

print("Testing CrossRNN forward:\n");
print(testCrossRNN:forward(sentenceTuple));

testCrossRNN:backward(sentenceTuple);
print("Testing CrossRNN backward success!\n");

testCrossRNN:updateParameters(learningRates);
print("Testing CrossRNN update success!\n");



