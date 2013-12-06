require 'torch'
require 'math'
require 'nn'

local function message(expected, actual)
  return 'Expected ' .. tostring(expected) .. ' but got ' .. tostring(actual)
end

function assertEquals(expected, actual)
  assert(expected == actual, message(expected, actual))
end

function assertFloatEquals(expected, actual, tolerance)
  if not tolerance then
    tolerance = 1e-8
  end
  assert(math.abs(expected - actual) < tolerance, message(expected, actual))
end

function assertTrue(actual)
  assertEquals(true, actual)
end

function assertFalse(actual)
  assertEquals(false, actual)
end
