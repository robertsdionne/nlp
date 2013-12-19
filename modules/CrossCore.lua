local CrossCore, parent = torch.class('nn.CrossCore', 'nn.Module')

function CrossCore:__init(initWeiht, initBias, tagId)
   parent.__init(self)
   self.tagId = tagId
   local outputSize = initWeiht:size()[1]
   local inputSize = initWeiht:size()[2]
   self.weight = initWeiht
   self.bias = initBias
   self.gradWeight = torch.rand(outputSize, inputSize)
   self.gradBias = torch.rand(outputSize)
   self.tanh = nn.Tanh()
end

function CrossCore:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.output:resize(nframe, nunit)
      self.output:zero():addr(1, input.new(nframe):fill(1), self.bias)
      self.output:addmm(1, input, self.weight:t())
   else
      error('input must be vector or matrix')
   end
   self.halfRes = self.output:clone()
   --print(self.output)
   self.output = self.tanh:forward(self.output)
   --self.output:add(1):mul(0.5)
--   print("Input of core")
   --print(input)
   --print(self.weight)
--   print("Output of core")
   --print(self.output)

   return self.output
end

function CrossCore:updateGradInput(input, gradOutput)
   if self.gradInput then
--      print("gradOutput for Core")
--      print(input)
--      print(self.output)
--      print(self.weight)
      --print(gradOutput)

      gradOutput = self.tanh:backward(self.halfRes, gradOutput)
      --gradOutput:mul(0.5)
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      end
      --print(gradOutput)
      --b = io.read()

--      print(self.gradInput)
      return self.gradInput
   end
end

function CrossCore:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   self.gradWeight:fill(0)
   self.gradBias:fill(0)
   gradOutput = self.tanh:backward(self.halfRes, gradOutput)
   --gradOutput:mul(0.5)
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      self.gradBias:add(scale, gradOutput)      
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nunit = self.bias:size(1)

      self.gradWeight:addmm(scale, gradOutput:t(), input)
      self.gradBias:addmv(scale, gradOutput:t(), input.new(nframe):fill(1))
   end
      --print("gradWeight for core")
      --print(self.gradWeight)
      --b = io.read()

end

-- Return the gradWeight {gradWeight, gradBias}
function CrossCore:getGradWeight()
   return {self.gradWeight, self.gradBias, tagId = self.tagId}
end
-- we do not need to accumulate parameters when sharing
CrossCore.sharedAccUpdateGradParameters = CrossCore.accUpdateGradParameters

