require "nn"
local CrossTagCVG, parent = torch.class('nn.CrossTagCVG', 'nn.Module')

--this module is used to get output from CROSSCORE and calculate tag and backPropagate
--It provides several basic functions: getOutPut, getGradWeight

--in this function, tag should be initialized in training. while in testing, tag should
--null or -1
function CrossTagCVG:__init(weight, bias, tag)
	-- body
	parent.__init(self)
	self.weight = weight
	self.bias = bias
	self.tag = tag
	-- self.predTag
	-- self.model
	-- self.probs
end

function CrossTagCVG:buildNet()
	local features = self.weight:size(2);
	local classes = self.weight:size(1);
	mlp = {}
	mlp = nn.Sequential()
	local linearLayer = nn.Linear(features, classes);
	linearLayer.weight = self.weight;
	linearLayer.bias = self.bias;
    linearLayer:zeroGradParameters()
	mlp:add(linearLayer)
    mlp:add(nn.Tanh())
	--:add( nn.LogSoftMax() )
	return mlp
end

--make sure call forwardBackward before getGradInput
--forwardBackward will calculate the value and store it as self.gradient
-- and getGradInput will simply return it
function CrossTagCVG:forward(input)
	self.model = self:buildNet();
	self.probs = self.model:forward(input)
	--print(self.probs);
	-- print(input);
	--print(self.tag);
	--io.read()
	_, self.predTag = self.probs:max(1)--@BUG
	self.predTag = self.predTag[1]
	self.score = self.probs[self.predTag]
end

function CrossTagCVG:backward(input)
	local criterion = nn.ClassNLLCriterion()
	local err = criterion:forward(self.probs, self.tag); 
	local t = criterion:backward(self.probs, self.tag);
	self.gradInput = self.model:backward(input, t);
	--updateParameters(0.3);
	self.weightGrad = self.model:get(1).gradWeight;
	self.biasGrad = self.model:get(1).gradBias;

end

--return the parameters of the current neurual network
function CrossTagCVG:getGradWeight()
	parameters = {self.weightGrad, self.biasGrad};
	return parameters;
end

function CrossTagCVG:getGradInput()
	return self.gradInput;
end


function CrossTagCVG:getPredTag()
	return self.predTag;
end
