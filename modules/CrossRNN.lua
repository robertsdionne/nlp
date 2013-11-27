require "nn"
local CrossRNN, parent = torch.class('nn.CrossRNN', 'nn.Module')

--build the RNN
function CrossRNN:__init(leftInputSize, rightInputSize, numTags, lookUpTable, initialNode)
-- init all parameters
	--self.paraIn
	--torch.Tensor(outputSize, inputSize)
	self.paraOut = {weight = torch.Tensor(numTags, leftInputSize), bias = torch.Tensor(numTags)}
	self.paraCore = {weight = torch.Tensor(leftInputSize,rightInputSize + leftInputSize),
			bias = torch.Tensor(leftInputSize)};
	self.initialNode = initialNode;
	self.gradients = {};	-- grads from each cross module
	--self.netWork	--stores the network
	--self.netWorkDepth		--stores the layer number of the network
end

--we assume that each sentence comes with tags
function CrossRNN:initializeCross(word, index, tag)
	inModule = nn.CrossWord(word, index);
	coreModule = nn.CrossCore(self.paraCore.weight, self.paraCore.bias);
	outModule = nn.CrossTag(self.paraOut.weight, self.paraOut.bias, tag);
	CrossModule = nn.Cross(coreModule, inModule, outModule);
	return CrossModule;
end

--the sentence tuple contains the sentence information, index information and the tag informtion
--the buildNet function will be call in forward. You have to make sure that forward
--is called before backward. This function will not be called in backward again.
function CrossRNN:buildNet(sentenceTuple)
	--print("here is the sentenceTuple");
	--print(sentenceTuple);

	self.netWorkDepth = #sentenceTuple.sentence;
	self.netWork = nn.Sequential();
	for i = 1, self.netWorkDepth do
		currentWord = sentenceTuple.sentence[i];
		currentIndex = sentenceTuple.index[i];
		currentTag = sentenceTuple.tag[i];
		self.netWork:add(self:initializeCross(currentWord,currentIndex,currentTag));
	end
end


function CrossRNN:forward(sentenceTuple)
	--print("sentence:\n");
	--print(sentenceTuple);
	-- unroll the RNN use sequentials
	self:buildNet(sentenceTuple);

	-- forward sequentialt for each of the cross module
	self.netWork:forward(self.initialNode);
	
	-- collect predicted tags
	predictedTags = {};
	for i = 1, self.netWorkDepth do
		predictedTags[i] = self.netWork:get(i).outModule:getPredTag();
	end
	
	-- return the predicted tags
	return predictedTags;
end

function CrossRNN:backward(sentenceTuple)
	
	--!!!need to becareful here that the final output/gradOutput of the sentence is null
	local finalGradOutput = torch.zeros(self.initialNode:size());
	-- backward the sequential
	self.netWork:backward(self.initialNode, finalGradOutput);

	-- collect gradParameters
	self.gradients = {};
	for i = 1, self.netWorkDepth do
		self.gradients[i] = self.netWork:get(i):getGradParameters();
	end
	--return gradients;
end

function CrossRNN:updateParameters(learningRates)
	--update the parameters
	local gradInWeightLength = self.gradients[1][1][1]:size();
	
	local gradCoreWeightLength = self.gradients[1][2][1]:size();
	--print("gradCoreWeightLength\n");
	--print(gradCoreWeightLength);
	local gradOutWeightLength = self.gradients[1][3][1]:size();

	--local gradInBiasLength = #self.gradients[1][1][2];
	local gradCoreBiasLength = #self.gradients[1][2][2];
	local gradOutBiasLength = #self.gradients[1][3][2];

	local gradInWeightSum = torch.Tensor(gradInWeightLength);
	local gradCoreWeightSum = torch.Tensor(gradCoreWeightLength);
		--print("xxxx\n");
	--print(gradCoreWeightSum);
	local gradOutWeightSum = torch.Tensor(gradOutWeightLength);
	--local gradInBiasSum = torch.zeros(gradInBiasLength);
	local gradCoreBiasSum = torch.Tensor(gradCoreBiasLength);
	local gradOutBiasSum = torch.Tensor(gradOutBiasLength);

	for i = 1, self.netWorkDepth do
		--print("self.gradients");
		--print(self.gradients);
		--print("gradInWeightSum\n");
		--print(gradInWeightSum);
		gradInWeightSum = gradInWeightSum + self.gradients[i][1][1];	--weight
		gradCoreWeightSum = gradCoreWeightSum + self.gradients[i][2][1];
		gradOutWeightSum = gradOutWeightSum + self.gradients[i][3][1];
		--gradInBiasSum = gradInBiasSum + self.gradients[i][1][2];
		gradCoreBiasSum = gradCoreBiasSum + self.gradients[i][2][2];
		gradOutBiasSum = gradOutBiasSum + self.gradients[i][3][2];
	end

	self.paraOut.weight = self.paraOut.weight - gradOutWeightSum * learningRates;
	self.paraOut.bias = self.paraOut.bias - gradOutBiasSum * learningRates;

	self.paraCore.weight = self.paraCore.weight - gradCoreWeightSum * learningRates;
	self.paraCore.bias = self.paraCore.bias - gradCoreBiasSum * learningRates;

	--not to be implemented here. Waiting for robert's function.
	--self.paraIn.weight = self.paraIn.weight - learningRates * gradInWeightSum;
	--self.paraIn.bias = self.paraIn.bias - learningRates * gradInBiasSum;
end