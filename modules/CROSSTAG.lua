require "nn"
local CROSSTAG, parent = torch.class('nn.CROSSTAG', 'nn.Module')

--this module is used to get output from CROSSCORE and calculate tag and backPropagate
--It provides several basic functions: getOutPut, getGradWeight

function CROSSTAG:__init(weight, bias, tag)
	-- body
	parent.__init(self)
	self.weight = weight
	self.bias = bias
	self.tag = tag
end

function CROSSTAG:buildNet()
	local features = self.weight.size(2);
	local classes = self.weight.size(1);
	mlp = {}
   	mlp = nn.Sequential()
   	mlp:add( nn.Linear(features, classes) )
   	mlp:add( nn.LogSoftMax() )
   	return mlp
end

--make sure call forwardBackward before getGradInput
--forwardBackward will calculate the value and store it as self.gradient
-- and getGradInput will simply return it
function forwardBackward(input)
	local model = CROSSTAG:buildNet();
	local criterion = nn.ClassNLLCriterion()
  	local pred = model:forward(input)
  	local err = criterion:forward(pred, self.tag); 
  	local t = criterion:backward(pred, self.tag);
  	self.gradient = model:backward(input, t);
end

--return the parameters of the current neurual network
function getGradWeight()
	parameters = {weight, bias};
	return parameters;
end

function getGradInput(input)
	return self.gradient;
end
