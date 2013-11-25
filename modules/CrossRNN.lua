local CrossRNN, parent = torch.class('nn.CrossRNN', 'nn.Module')

--build the RNN
function CrossRNN:__init(leftInputSize, rightInputSize, numTags, lookUpTable)
-- init all parameters
-- self.paraIn
-- self.paraOut
-- self.paraCore
self.gradIns = {} -- grads from inMoudles
self.gradOuts = {} -- grads from outMoudles
self.gradCores = {} -- grads from coreMoudles
end

function CrossRNN:forward(sentence)
-- unroll the RNN use sequential
-- forward sequential
-- collect predicted tags
-- return the predicted tags
end

function CrossRNN:backward(sentence)
-- backward the sequential
-- collect gradParameters

end

function CrossRNN:updateParameters(learningRates)
--update the parameters
end