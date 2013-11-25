require "nn"
local CrossWord, parent = torch.class('nn.CrossWord', 'nn.Module')

--this module is used to store the word index and its representation.
--It provides several basic functions: getOutPut, getGradWeight

function CrossWord:__init(inputWord, inputIndex)
	-- body
	parent.__init(self)
	self.inputWord = inputWord
	self.inputIndex = inputIndex
end

--simply return the word representation to the upper level
function  CrossWord:getOutput()
	-- body
	return self.inputWord;
end

--calculate the gradient for lower level unit, input is the gradient from upper level
--actually just combining the gradient from upper level and the word index
function CrossWord:getGradWeight(input)
	return {input, inputIndex}
end