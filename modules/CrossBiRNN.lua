dofile "CrossRNN.lua"

local CrossBiRNN, parent = torch.class('nn.CrossBiRNN', 'nn.Module')

--build the RNN
function CrossBiRNN:__init(leftInputSize, rightInputSize, numTags, lookUpTable)
	self.forwardRNN = nn.CrossRNN(leftInputSize, rightInputSize, numTags, lookUpTable)
	self.backwardRNN = nn.CrossRNN(leftInputSize, rightInputSize, numTags, lookUpTable)
end

--we assume that each sentence comes with tags
function CrossBiRNN:inverseSent(sentenceTuple)

end

function CrossBiRNN:forward(sentenceTuple, initialNode)
	-- get inverse sentence for backward
	-- represents index tagsId
	local invSentenceTuple = {represents = {}, index = {}, tagsId = {}}
	
	-- forward both models

	-- collect probs from two models

	-- analysis to get jointly predicted tags
	
	-- return the predicted tags
	return predictedTags;
end

function CrossBiRNN:backward(sentenceTuple, initialNode)
	
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

function CrossBiRNN:updateCoreParameters(learningRates)
	
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

function CrossBiRNN:updateOutParameters(learningRates)
	
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

function CrossBiRNN:updateInParameters(learningRates)
	
	local gradInWeightLength = self.gradients[1][1][1]:size();
	local gradInWeightSum = torch.rand(gradInWeightLength):fill(0);
	
	for i = 1, self.netWorkDepth do
		wordIndex = self.netWork:get(i).inModule.inputIndex;
		wordGradient = torch.Tensor(1, self.gradients[i][1][1]:size(1)):copy(self.gradients[i][1][1]);
		--self.lookUpTable:backwardUpdate(wordIndex, wordGradient, 0.001);
	end
	-- update the initialNode
	initialNodeGrad = torch.Tensor(1,self.gradients[1][1][1]:size(1)):copy(self.initialNodeGrad);
	-- self.lookUpTable:backwardUpdate('PADDING', initialNodeGrad, learningRates);
end


function CrossBiRNN:updateParameters(learningRates)
	--update the parameters

    self:updateCoreParameters(learningRates)
    self:updateOutParameters(learningRates)
    -- self:updateInParameters(learningRates)

	
end
