require "nn"
local CrossTag, parent = torch.class('nn.CrossTag', 'nn.Module')

--this module is used to get output from CROSSCORE and calculate tag and backPropagate
--It provides several basic functions: getOutPut, getGradWeight

function CrossTag:__init(weight, bias, tag)
	-- body
	parent.__init(self)
	self.weight = weight
	self.bias = bias
	self.tag = tag
end

function CrossTag:buildNet()
	local features = self.weight:size(2);
	local classes = self.weight:size(1);
	mlp = {}
	mlp = nn.Sequential()
	local linearLayer = nn.Linear(features, classes);
	linearLayer.weight = self.weight;
	linearLayer.bias = self.bias;
	mlp:add(linearLayer);
	mlp:add( nn.LogSoftMax() )
	return mlp
end

--make sure call forwardBackward before getGradInput
--forwardBackward will calculate the value and store it as self.gradient
-- and getGradInput will simply return it
function CrossTag:forwardBackward(input)
	local model = self:buildNet();
	local criterion = nn.ClassNLLCriterion()
	local pred = model:forward(input)
	local err = criterion:forward(pred, self.tag); 
	local t = criterion:backward(pred, self.tag);
	self.gradInput = model:backward(input, t);
	self.weightGrad = model:get(1).gradWeight;
	self.biasGrad = model:get(1).gradBias;
end

--return the parameters of the current neurual network
function CrossTag:getGradWeight()
	parameters = {self.weightGrad, self.biasGrad};
	return parameters;
end

function CrossTag:getGradInput(input)
	return self.gradInput;
end
