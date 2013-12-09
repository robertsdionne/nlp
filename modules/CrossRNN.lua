require "nn"
dofile "CrossWord.lua"
dofile "CrossTag.lua"
dofile "CrossCore.lua"
dofile "Cross.lua"
local CrossRNN, parent = torch.class('nn.CrossRNN', 'nn.Module')

--build the RNN
function CrossRNN:__init(leftInputSize, rightInputSize, numTags, lookUpTable)
-- init all parameters
	--self.paraIn
	--torch.Tensor(outputSize, inputSize)
	self.paraOut = {weight = torch.rand(numTags, leftInputSize), bias = torch.rand(numTags)}
	self.paraCore = {weight = torch.rand(leftInputSize,rightInputSize + leftInputSize),
			bias = torch.rand(leftInputSize)};
	--print("the initial core weight:\n");
	--print(self.paraCore.weight);
	self.lookUpTable = lookUpTable;
	self.gradients = {};	-- grads from each cross module
	--self.initialNodeGrad		--The gradient of the initialNode (the returned valud of self.netWork:backward() )
	--self.netWork	--stores the network
	--self.netWorkDepth		--stores the layer number of the network
end

--we assume that each sentence comes with tags
function CrossRNN:initializeCross(word, index, tagId)
	inModule = nn.CrossWord(word, index);
	coreModule = nn.CrossCore(self.paraCore.weight, self.paraCore.bias);
	outModule = nn.CrossTag(self.paraOut.weight, self.paraOut.bias, tagId);
	CrossModule = nn.Cross(coreModule, inModule, outModule);
	return CrossModule;
end

--the sentence tuple contains the sentence information, index information and the tag informtion
--the buildNet function will be call in forward. You have to make sure that forward
--is called before backward. This function will not be called in backward again.
function CrossRNN:buildNet(sentenceTuple)
	self.netWorkDepth = #sentenceTuple.represents;
	self.netWork = nn.Sequential();
	for i = 1, self.netWorkDepth do
		currentWord = sentenceTuple.represents[i];
		--print("currentWord")
		--print(currentWord)
		currentIndex = sentenceTuple.index[i];
		currentTagId = sentenceTuple.tagsId[i];
		self.netWork:add(self:initializeCross(currentWord,currentIndex,currentTagId));
	end
end


function CrossRNN:forward(sentenceTuple, initialNode)
	-- unroll the RNN use sequentials
	self:buildNet(sentenceTuple);

	-- forward sequentialt for each of the cross module
	self.netWork:forward(initialNode);
	
	-- collect predicted tags
	predictedTags = {};
	for i = 1, self.netWorkDepth do
		predictedTags[i] = self.netWork:get(i).outModule:getPredTag();
	end
	
	-- return the predicted tags
	return predictedTags;
end

function CrossRNN:backward(sentenceTuple, initialNode)
	
	--!!!need to becareful here that the final output/gradOutput of the sentence is null
	local finalGradOutput = torch.zeros(initialNode:size());
	-- backward the sequential
	self.initialNodeGrad = self.netWork:backward(initialNode, finalGradOutput);

	-- collect gradParameters
	self.gradients = {};
	for i = 1, self.netWorkDepth do
		self.gradients[i] = self.netWork:get(i):getGradParameters();
        --print("Gradients")
        --print(self.gradients[i][2][1])
        --b = io.read()
	end
	--return gradients;
end

function CrossRNN:updateParameters(learningRates)
	--update the parameters
	local gradInWeightLength = self.gradients[1][1][1]:size();
	
	local gradCoreWeightLength = self.gradients[1][2][1]:size();
	local gradOutWeightLength = self.gradients[1][3][1]:size();

	local gradCoreBiasLength = #self.gradients[1][2][2];
	local gradOutBiasLength = #self.gradients[1][3][2];

	local gradInWeightSum = torch.rand(gradInWeightLength):fill(0);
	local gradCoreWeightSum = torch.rand(gradCoreWeightLength):fill(0);
	local gradOutWeightSum = torch.rand(gradOutWeightLength):fill(0);
	local gradCoreBiasSum = torch.rand(gradCoreBiasLength):fill(0);
	local gradOutBiasSum = torch.rand(gradOutBiasLength):fill(0);
	--print(gradOutWeightSum)
	for i = 1, self.netWorkDepth do
		--call Roberts function to update word representation.
		--this is actually updating the InParas(words)
		wordIndex = self.netWork:get(i).inModule.inputIndex;
		wordGradient = torch.Tensor(1, self.gradients[i][1][1]:size(1)):copy(self.gradients[i][1][1]);
		--self.lookUpTable:backwardUpdate(wordIndex, wordGradient, 0.001);

    --print("gradOutWeightSum  "..i)
    --print(gradOutWeightSum)
    --print(self.gradients[i][3][1])
    --io.read()
		--get the sum of all the gradients
		gradCoreWeightSum = gradCoreWeightSum + self.gradients[i][2][1];
		gradOutWeightSum = gradOutWeightSum + self.gradients[i][3][1];
		gradCoreBiasSum = gradCoreBiasSum + self.gradients[i][2][2];
		gradOutBiasSum = gradOutBiasSum + self.gradients[i][3][2];
	end
    --print(torch.norm(gradOutWeightSum))
    --print("gradOutWeightSum")
    --print(gradOutWeightSum)
    --io.read()
    --b = io.read()

	--update the weight matrix parameters
	self.paraOut.weight = self.paraOut.weight - gradOutWeightSum * learningRates;
	self.paraOut.bias = self.paraOut.bias - gradOutBiasSum * learningRates;

	self.paraCore.weight = self.paraCore.weight - gradCoreWeightSum * learningRates;
	self.paraCore.bias = self.paraCore.bias - gradCoreBiasSum * learningRates;

	--update the initialNode
	initialNodeGrad = torch.Tensor(1,self.gradients[1][1][1]:size(1)):copy(self.initialNodeGrad);
	--self.lookUpTable:backwardUpdate('PADDING', initialNodeGrad, learningRates);
end
