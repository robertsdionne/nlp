require 'math'

local function message(expected, actual)
  return 'Expected ' .. tostring(expected) .. ' but got ' .. tostring(actual)
end

function assertEquals(expected, actual)
  assert(expected == actual, message(expected, actual))
end

function assertFalse(actual)
  assertEquals(false, actual)
end

function assertFloatEquals(expected, actual, tolerance)
  tolerance = tolerance or 1e-8
  assert(math.abs(expected - actual) < tolerance, message(expected, actual))
end

function assertNil(actual)
  assertEquals(nil, actual)
end

function assertNotNil(actual)
  assertFalse(nil == actual)
end

function assertTrue(actual)
  assertEquals(true, actual)
end
