require "nn"

dofile "CrossWord.lua"
dofile "CrossTag.lua"
dofile "CrossCore.lua"
-- dofile "CrossCoreFeak.lua"
dofile "Cross.lua"
dofile "Weights.lua"

local CrossRNN, parent = torch.class('nn.CrossRNN', 'nn.Module')

--build the RNN
function CrossRNN:__init(leftInputSize, rightInputSize, numTags, lookUpTable)
-- init all parameters
	--self.paraIn
	--torch.Tensor(outputSize, inputSize)
	self.paraOut = {
		weight = nn.Weights.normalizedInitializationSigmoid(numTags, leftInputSize),
		bias = nn.Weights.zeros(numTags)
	}
	self.paraCore = {
		weight = nn.Weights.normalizedInitializationTanh(leftInputSize, rightInputSize + leftInputSize),
		bias = nn.Weights.zeros(leftInputSize)
	};
	--print("the initial core weight:\n");
	--print(self.paraCore.weight);
	self.lookUpTable = lookUpTable;
	self.gradients = {};	-- grads from each cross module
	self.adaLearningRates = {};	-- grads from each cross module
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
		self.netWork:add(self:initializeCross(currentWord, currentIndex, currentTagId));
	end
end


function CrossRNN:forward(sentenceTuple, initialNode)
	-- unroll the RNN use sequentials
	self:buildNet(sentenceTuple)
	-- forward sequentialt for each of the cross module
	self.netWork:forward(initialNode)
	
	-- collect predicted tags
	predictedTags = {};
	for i = 1, self.netWorkDepth do
		predictedTags[i] = self.netWork:get(i).outModule:getPredTag();
	end
	
	-- return the predicted tags
	return predictedTags;
end

function CrossRNN:getProbs()
	local probs = {}
	for i = 1, self.netWorkDepth do
		table.insert(probs, self.netWork:get(i).outModule.probs)
	end
	return probs
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

function CrossRNN:updateCoreParameters(learningRates)
	
	local gradCoreWeightLength = self.gradients[1][2][1]:size();
	local gradCoreBiasLength = #self.gradients[1][2][2];
	-- local gradCoreWeightSum = torch.rand(gradCoreWeightLength):fill(0);
	-- local gradCoreBiasSum = torch.rand(gradCoreBiasLength):fill(0);

	if self.adaLearningRates.gradCoreWeightLR == nil then
		self.adaLearningRates.gradCoreWeightLR = torch.zeros(gradCoreWeightLength);
		self.adaLearningRates.gradCoreBiasLR = torch.zeros(gradCoreBiasLength);
	end



	for i = 1, self.netWorkDepth do
		--get the sum of all the gradients
		-- gradCoreWeightSum = gradCoreWeightSum + self.gradients[i][2][1];
		-- gradCoreBiasSum = gradCoreBiasSum + self.gradients[i][2][2];
		self.adaLearningRates.gradCoreWeightLR = self.adaLearningRates.gradCoreWeightLR + torch.cmul(self.gradients[i][2][1],self.gradients[i][2][1]);
		self.adaLearningRates.gradCoreBiasLR = self.adaLearningRates.gradCoreBiasLR + torch.cmul(self.gradients[i][2][2],self.gradients[i][2][2]);
	end

	for i = 1, self.netWorkDepth do
		--get the sum of all the gradients
		self.paraCore.weight = self.paraCore.weight - torch.cdiv(self.gradients[i][2][1], torch.sqrt(self.adaLearningRates.gradCoreWeightLR))  * learningRates;
		self.paraCore.bias = self.paraCore.bias - torch.cdiv(self.gradients[i][2][2], torch.sqrt(self.adaLearningRates.gradCoreBiasLR))  * learningRates;
	end

	
	
	-- self.paraCore.weight = self.paraCore.weight - gradCoreWeightSum * learningRates;
	-- self.paraCore.bias = self.paraCore.bias - gradCoreBiasSum * learningRates;
end

function CrossRNN:updateOutParameters(learningRates)
	
	local gradOutWeightLength = self.gradients[1][3][1]:size();
	local gradOutBiasLength = #self.gradients[1][3][2];

	-- local gradOutWeightSum = torch.rand(gradOutWeightLength):fill(0);
	-- local gradOutBiasSum = torch.rand(gradOutBiasLength):fill(0);


	if self.adaLearningRates.gradOutWeightLR == nil then
		self.adaLearningRates.gradOutWeightLR = torch.rand(gradOutWeightLength):fill(0);
		self.adaLearningRates.gradOutBiasLR = torch.rand(gradOutBiasLength):fill(0);
	end

	for i = 1, self.netWorkDepth do
		--get the sum of all the gradients
		-- gradOutWeightSum = gradOutWeightSum + self.gradients[i][3][1];
		-- gradOutBiasSum = gradOutBiasSum + self.gradients[i][3][2];

		self.adaLearningRates.gradOutWeightLR = self.adaLearningRates.gradOutWeightLR + torch.cmul(self.gradients[i][3][1],self.gradients[i][3][1]);
		self.adaLearningRates.gradOutBiasLR = self.adaLearningRates.gradOutBiasLR + torch.cmul(self.gradients[i][3][2],self.gradients[i][3][2]);
	end

	for i = 1, self.netWorkDepth do
		--get the sum of all the gradients

		self.paraOut.weight = self.paraOut.weight - torch.cdiv(self.gradients[i][3][1], torch.sqrt(self.adaLearningRates.gradOutWeightLR))  * learningRates;
		self.paraOut.bias = self.paraOut.bias - torch.cdiv(self.gradients[i][3][2], torch.sqrt(self.adaLearningRates.gradOutBiasLR))  * learningRates;
	end

	
	
	-- self.paraOut.weight = self.paraOut.weight - gradOutWeightSum * learningRates;
	-- self.paraOut.bias = self.paraOut.bias - gradOutBiasSum * learningRates;
end

function CrossRNN:updateInParameters(learningRates)
	
	local gradInWeightLength = self.gradients[1][1][1]:size();
	local gradInWeightSum = torch.rand(gradInWeightLength):fill(0);
	
	for i = 1, self.netWorkDepth do
		wordIndex = self.netWork:get(i).inModule.inputIndex;
		wordGradient = torch.Tensor(1, self.gradients[i][1][1]:size(1)):copy(self.gradients[i][1][1]);
		-- self.lookUpTable:backwardUpdate(wordIndex, wordGradient, 0.001);
	end
	-- update the initialNode
	initialNodeGrad = torch.Tensor(1,self.gradients[1][1][1]:size(1)):copy(self.initialNodeGrad);
	-- self.lookUpTable:backwardUpdate('PADDING', initialNodeGrad, learningRates);
end


function CrossRNN:updateParameters(learningRates)
	--update the parameters

    self:updateCoreParameters(learningRates)
    self:updateOutParameters(learningRates)
    self:updateInParameters(learningRates)

	
end
